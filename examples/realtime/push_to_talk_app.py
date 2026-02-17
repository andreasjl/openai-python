#!/usr/bin/env uv run
####################################################################
# Sample TUI app with a push to talk interface to the Realtime API #
# If you have `uv` installed and the `AZURE_OPENAI_ENDPOINT` set   #
# to your AOAI endpoint from Azure Portal                          #
# (e.g., https://YOUR_ACCOUNT.openai.azure.com/), and the          #
# `AZURE_OPENAI_DEPLOYMENT` set to your deployment name from       #
# AI Foundry.                                                      #
# The application uses Entra ID authentication with                #
# DefaultAzureCredentials. Make sure to have at least the          #
# "Cognitive Services OpenAI User" role assigned to your account.  #
# environment variable set, you can run this example with just     #
#                                                                  #
# `./examples/realtime/push_to_talk_app.py`                        #
#                                                                  #
# On Mac, you'll also need `brew install portaudio ffmpeg`         #
####################################################################
#
# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "textual",
#     "numpy",
#     "pyaudio",
#     "pydub",
#     "sounddevice",
#     "openai[realtime]",
#     "openai",
#     "azure-identity",
#     "aiohttp"
# ]
#
# [tool.uv.sources]
# openai = { path = "../../", editable = true }
# ///
from __future__ import annotations

import base64
import asyncio
from typing import Any, cast
from typing_extensions import override

from textual import events
from audio_util import CHANNELS, SAMPLE_RATE, AudioPlayerAsync
from textual.app import App, ComposeResult
from textual.widgets import Button, Static, RichLog
from textual.reactive import reactive
from textual.containers import Container

import os

from azure.identity.aio import DefaultAzureCredential, get_bearer_token_provider

from openai import AsyncAzureOpenAI, AsyncOpenAI
from openai.types.beta.realtime.session import Session
from openai.resources.beta.realtime.realtime import AsyncRealtimeConnection

class SessionDisplay(Static):
    """A widget that shows the current session ID."""

    session_id = reactive("")

    @override
    def render(self) -> str:
        return f"Session ID: {self.session_id}" if self.session_id else "Connecting..."


class AudioStatusIndicator(Static):
    """A widget that shows the current audio recording status."""

    is_recording = reactive(False)

    @override
    def render(self) -> str:
        status = (
            "🔴 Recording... (Press K to stop)" if self.is_recording else "⚪ Press K to start recording (Q to quit)"
        )
        return status


class RealtimeApp(App[None]):
    CSS = """
        Screen {
            background: #1a1b26;  /* Dark blue-grey background */
        }

        Container {
            border: double rgb(91, 164, 91);
        }

        Horizontal {
            width: 100%;
        }

        #input-container {
            height: 5;  /* Explicit height for input container */
            margin: 1 1;
            padding: 1 2;
        }

        Input {
            width: 80%;
            height: 3;  /* Explicit height for input */
        }

        Button {
            width: 20%;
            height: 3;  /* Explicit height for button */
        }

        #bottom-pane {
            width: 100%;
            height: 82%;  /* Reduced to make room for session display */
            border: round rgb(205, 133, 63);
            content-align: center middle;
        }

        #status-indicator {
            height: 3;
            content-align: center middle;
            background: #2a2b36;
            border: solid rgb(91, 164, 91);
            margin: 1 1;
        }

        #session-display {
            height: 3;
            content-align: center middle;
            background: #2a2b36;
            border: solid rgb(91, 164, 91);
            margin: 1 1;
        }

        Static {
            color: white;
        }
    """

    client: AsyncOpenAI #AsyncAzureOpenAI
    should_send_audio: asyncio.Event
    audio_player: AudioPlayerAsync
    last_audio_item_id: str | None
    connection: AsyncRealtimeConnection | None
    session: Session | None
    connected: asyncio.Event

    def __init__(self) -> None:
        super().__init__()
        self.connection = None
        self.session = None

        # self.client = AsyncAzureOpenAI(
        #     azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        #     azure_ad_token_provider=get_bearer_token_provider(DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"),
        #     api_version="2024-10-01-preview"
        # )
        self.client = AsyncOpenAI(
            api_key="TODO"
        )

        self.audio_player = AudioPlayerAsync()
        self.last_audio_item_id = None
        self.should_send_audio = asyncio.Event()
        self.connected = asyncio.Event()

    @override
    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        with Container():
            yield SessionDisplay(id="session-display")
            yield AudioStatusIndicator(id="status-indicator")
            yield RichLog(id="bottom-pane", wrap=True, highlight=True, markup=True)

    async def on_mount(self) -> None:
        self.run_worker(self.handle_realtime_connection())
        self.run_worker(self.send_mic_audio())

    async def handle_realtime_connection(self) -> None:
        async with self.client.beta.realtime.connect(
            # model=os.environ["AZURE_OPENAI_DEPLOYMENT"],
            model = "gpt-realtime-2026-01-12"
            # extra_headers={
            #     "x-ms-oai-assistants-testenv": "dev.hyena",
            #     "x-ms-oai-assistants-testdep": "pull-63993",
            # },
        ) as conn:
            self.connection = conn
            self.connected.set()

            # note: this is the default and can be omitted
            # if you want to manually handle VAD yourself, then set `'turn_detection': None`
            await conn.session.update(session={"modalities": ["audio", "text"], "voice": "cedar", "instructions": "You are a helpful AI assistant. Respond naturally and conversationally.", "turn_detection": {"type": "server_vad"}, "input_audio_transcription": { "model": "gpt-4o-transcribe"}})

            async for event in conn:
                bottom_pane = self.query_one("#bottom-pane", RichLog)
                bottom_pane.write(event.type)
                if event.type == "session.created":
                    self.session = event.session
                    session_display = self.query_one(SessionDisplay)
                    assert event.session.id is not None
                    session_display.session_id = event.session.id
                    continue

                if event.type == "session.updated":
                    self.session = event.session
                    bottom_pane.write(f"Session updated: {event}")
                    continue

                if event.type == "error":
                    bottom_pane.write(f"Error: {event}")
                    continue
                if event.type == "response.done" or event.type == "response.output_item.done":
                    bottom_pane.write(f"{event.type}: {event}")
                    continue
                if event.type == "conversation.item.created":
                    bottom_pane.write(f"Conversation item created: {event}")
                    continue


                if event.type == "response.audio.delta":
                    if event.item_id != self.last_audio_item_id:
                        self.audio_player.stop()
                        self.audio_player.reset_frame_count()
                        self.last_audio_item_id = event.item_id

                    bytes_data = base64.b64decode(event.delta)
                    self.audio_player.add_data(bytes_data)
                    continue

                if event.type == "conversation.item.input_audio_transcription.completed":
                    bottom_pane.write(f"[bold green]Transcription:[/bold green] {event}")
                    continue


    async def _get_connection(self) -> AsyncRealtimeConnection:
        await self.connected.wait()
        assert self.connection is not None
        return self.connection

    async def send_mic_audio(self) -> None:
        import sounddevice as sd  # type: ignore

        sent_audio = False

        device_info = sd.query_devices()
        print(device_info)

        read_size = int(SAMPLE_RATE * 0.02)

        stream = sd.InputStream(
            channels=CHANNELS,
            samplerate=SAMPLE_RATE,
            dtype="int16",
        )
        stream.start()

        status_indicator = self.query_one(AudioStatusIndicator)

        try:
            while True:
                if stream.read_available < read_size:
                    await asyncio.sleep(0)
                    continue

                await self.should_send_audio.wait()
                status_indicator.is_recording = True

                data, _ = stream.read(read_size)

                connection = await self._get_connection()
                if not sent_audio:
                    asyncio.create_task(connection.send({"type": "response.cancel"}))
                    sent_audio = True

                await connection.input_audio_buffer.append(audio=base64.b64encode(cast(Any, data)).decode("utf-8"))

                await asyncio.sleep(0)
        except KeyboardInterrupt:
            pass
        finally:
            stream.stop()
            stream.close()

    async def on_key(self, event: events.Key) -> None:
        """Handle key press events."""
        if event.key == "enter":
            self.query_one(Button).press()
            return

        if event.key == "q":
            self.exit()
            return

        if event.key == "k":
            status_indicator = self.query_one(AudioStatusIndicator)
            if status_indicator.is_recording:
                self.should_send_audio.clear()
                status_indicator.is_recording = False

                if self.session and self.session.turn_detection is None:
                    # The default in the API is that the model will automatically detect when the user has
                    # stopped talking and then start responding itself.
                    #
                    # However if we're in manual `turn_detection` mode then we need to
                    # manually tell the model to commit the audio buffer and start responding.
                    conn = await self._get_connection()
                    await conn.input_audio_buffer.commit()
                    await conn.response.create()
            else:
                self.should_send_audio.set()
                status_indicator.is_recording = True


def handle_exception(exc_type, exc_value, exc_traceback):
    print("An unhandled exception occurred:", exc_value)
    print("Exception type:", exc_type)
    print("Traceback:", exc_traceback)

if __name__ == "__main__":
    import sys

    # Set the exception handler to catch unhandled exceptions
    sys.excepthook = handle_exception
    app = RealtimeApp()
    app.run()
