import asyncio
import logging
from dataclasses import dataclass
from typing import AsyncIterable, Optional
import aiohttp
from livekit import rtc
from livekit.agents import tts, utils

API_BASE_URL = "http://10.0.0.119:6666"  # Replace with your API base URL


@dataclass
class TTSOptions:
    base_url: str
    sample_rate: int


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        sample_rate: int = 24000,
    ) -> None:
        super().__init__(streaming_supported=True)
        self._session = aiohttp.ClientSession()
        self._config = TTSOptions(
            base_url=base_url or API_BASE_URL,
            sample_rate=sample_rate,
        )

    def synthesize(
        self,
        *,
        text: str,
    ) -> AsyncIterable[tts.SynthesizedAudio]:
        results = utils.AsyncIterableQueue()

        async def fetch_task():
            async with self._session.get(
                f"{self._config.base_url}/api/tts-stream",
                params={"text": text},
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
                logging.error(f"TTS synthesis task failed: {task.exception()}")

        self._main_task.add_done_callback(log_exception)
        self._text = ""

    def push_text(self, token: str) -> None:
        if self._closed:
            raise ValueError("cannot push to a closed stream")

        if not token or len(token) == 0:
            return

        self._text += token
        self._queue.put_nowait(self._text)
        self._text = ""

    async def _run(self) -> None:
        while True:
            text = await self._queue.get()
            self._queue.task_done()

            async with self._session.get(
                f"{self._config.base_url}/api/tts-stream",
                params={"text": text},
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

                self._event_queue.put_nowait(
                    tts.SynthesisEvent(type=tts.SynthesisEventType.FINISHED)
                )

    async def flush(self) -> None:
        self._queue.put_nowait(self._text)
        self._text = ""
        await self._queue.join()

    async def aclose(self) -> None:
        self._main_task.cancel()
        try:
            await self._main_task
        except asyncio.CancelledError:
            pass

    async def __anext__(self) -> tts.SynthesisEvent:
        if self._closed and self._event_queue.empty():
            raise StopAsyncIteration

        return await self._event_queue.get()                