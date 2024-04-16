import asyncio
import logging
from dataclasses import dataclass
from typing import AsyncIterable, Optional
import aiohttp
from livekit import rtc
from livekit.agents import tts, utils
import io
import wave
import base64
import contextlib

API_BASE_URL = "http://10.0.0.119:6666"
DEFAULT_VOICE = "goldvoice"

@dataclass
class TTSOptions:
    voice: str
    base_url: str
    sample_rate: int


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        voice: str = DEFAULT_VOICE,
        base_url: Optional[str] = None,
        sample_rate: int = 24000,
    ) -> None:
        super().__init__(streaming_supported=True)
        self._session = aiohttp.ClientSession()
        self._config = TTSOptions(
            voice=voice,
            base_url=base_url or API_BASE_URL,
            sample_rate=sample_rate,
        )
    
    def set_voice(self, voice: str) -> None:
        self._config.voice = voice
    
    async def upload_audio(self, file_name: str, buffer: list[rtc.AudioFrame]) -> None:
        print("uploading audio")
        # Merge the audio frames in the buffer
        merged_frame = utils.merge_frames(buffer)

        # Create a BytesIO object and write the audio data as a WAV file
        io_buffer = io.BytesIO()
        with wave.open(io_buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)  # 16-bit
            wav.setframerate(16000)
            wav.writeframes(merged_frame.data)

        # Reset the BytesIO object to the beginning
        io_buffer.seek(0)

        # Read the audio data and encode it in base64
        audio_data = base64.b64encode(io_buffer.read()).decode('utf-8')

        # Create the data dictionary for the POST request
        data = {
            "file_name": file_name,
            "audio_data": audio_data
        }

        # Send the POST request to upload the audio file
        async with self._session.post(
            f"{self._config.base_url}/api/upload-audio",
            data=data,
        ) as resp:
            # Check the response status
            if resp.status != 200:
                logging.error(f"Failed to upload audio file: {resp.status}")

    def synthesize(
        self,
        *,
        text: str,
    ) -> AsyncIterable[tts.SynthesizedAudio]:
        results = utils.AsyncIterableQueue()

        async def fetch_task():
            async with self._session.get(
                f"{self._config.base_url}/api/tts-cloned-stream",
                params={
                    "text": text,
                    "language": "en",
                    "style-wav": self._config.voice,
                },
            ) as resp:
                data = await resp.read()
                results.put_nowait(
                    tts.SynthesizedAudio(
                        text=text,
                        data=rtc.AudioFrame(
                            data=data,
                            sample_rate=self._config.sample_rate,
                            num_channels=1,
                            samples_per_channel=len(data) // 2,  # 16-bit
                        ),
                    )
                )
                results.close()

        asyncio.ensure_future(fetch_task())

        return results

    def stream(
        self,
    ) -> tts.SynthesizeStream:
        return SynthesizeStream(self._session, self._config)


class SynthesizeStream(tts.SynthesizeStream):
    def __init__(
        self,
        session: aiohttp.ClientSession,
        config: TTSOptions,
    ):
        self._config = config
        self._session = session
        self._queue = asyncio.Queue[str]()
        self._event_queue = asyncio.Queue[tts.SynthesisEvent]()
        self._closed = False
        self._main_task = asyncio.create_task(self._run())

        def log_exception(task: asyncio.Task) -> None:
            if not task.cancelled() and task.exception():
                logging.error(f"XTTS TTS synthesis task failed: {task.exception()}")

        self._main_task.add_done_callback(log_exception)
        self._text = ""

    # TODO make this streaming
    def push_text(self, token: str) -> None:
        if self._closed:
            raise ValueError("cannot push to a closed stream")

        if not token or len(token) == 0:
            return

        self._text += token

    async def _run(self) -> None:
        try:
            text = await self._queue.get()
            self._queue.task_done()

            async with self._session.get(
                f"{self._config.base_url}/api/tts-cloned-stream",
                params={
                    "text": text,
                    "language": "en",
                    "style-wav": self._config.voice,
                },
            ) as resp:
                self._event_queue.put_nowait(
                    tts.SynthesisEvent(type=tts.SynthesisEventType.STARTED)
                )

                async for chunk in resp.content.iter_chunked(4096):
                    audio_frame = rtc.AudioFrame(
                        data=chunk,
                        sample_rate=self._config.sample_rate,
                        num_channels=1,
                        samples_per_channel=len(chunk) // 2,  # 16-bit
                    )
                    self._event_queue.put_nowait(
                        tts.SynthesisEvent(
                            type=tts.SynthesisEventType.AUDIO,
                            audio=tts.SynthesizedAudio(text=text, data=audio_frame),
                        )
                    )
                print("putting finished")
                self._event_queue.put_nowait(
                    tts.SynthesisEvent(type=tts.SynthesisEventType.FINISHED)
                )

        except asyncio.CancelledError:
            raise

        finally:
            await self._session.close()
            self._closed = True

    async def flush(self) -> None:
        self._queue.put_nowait(self._text)
        self._text = ""
        await self._queue.join()

    async def aclose(self) -> None:
        self._main_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._main_task
            print("TTS CLOSED")

    async def __anext__(self) -> tts.SynthesisEvent:
        if self._closed and self._event_queue.empty():
            raise StopAsyncIteration

        return await self._event_queue.get()                