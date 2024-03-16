# Copyright 2023 Purfect, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import base64
import io
import json
import logging
from dataclasses import dataclass
from typing import Optional, Set

import aiohttp
from livekit import rtc
from PIL import Image

class OllamaMultiModal:
    """Fal SDXL Plugin

    Requires FAL_KEY_ID and FAL_KEY_SECRET environment variables to be set.
    """

    def __init__(
        self,
    ):
        self._url = "http://localhost:11434/api/generate"

    def stream(self) -> "OllamaMutliModalStream":
        return OllamaMutliModalStream(url=self._url)


def _task_done_cb(task: asyncio.Task, set: Set[asyncio.Task]) -> None:
    set.discard(task)
    if task.cancelled():
        logging.info("task cancelled: %s", task)
        return

    if task.exception():
        logging.error("task exception: %s", task, exc_info=task.exception())
        return


class OllamaMutliModalStream:
    @dataclass
    class Input:
        data: bytes
        prompt: str
        width: int
        height: int

    def __init__(self, url: str):
        self._url = url
        self._input_queue = asyncio.Queue[self.Input]()
        self._output_queue = asyncio.Queue[str]()
        self._closed = False
        self._run_task = asyncio.create_task(self._run())

    def push_frame(self, frame: rtc.VideoFrame, prompt: str) -> None:
        if self._closed:
            raise ValueError("cannot push to a closed stream")

        argb_frame = frame.convert(rtc.VideoBufferType.RGBA)
        self._input_queue.put_nowait(
            self.Input(
                data=argb_frame.data,
                prompt=prompt,
                width=frame.width,
                height=frame.height,
            )
        )

    async def aclose(self) -> None:
        self._run_task.cancel()
        await self._run_task
        self._closed = True

    async def _run(self):
        async with aiohttp.ClientSession() as session:
            while True:
                try:
                    input = await self._input_queue.get()
                    # print(f"\nReceived input: {input}\n")

                    image_rgba = Image.frombytes(
                        "RGBA", (input.width, input.height), input.data
                    )
                    image_rgb = image_rgba.convert("RGB")
                    # center crop to 512x512
                    # crop_x = (image_rgb.width - 512) // 2
                    # crop_y = (image_rgb.height - 512) // 2
                    # image_rgb = image_rgb.crop(
                    #     (crop_x, crop_y, crop_x + 512, crop_y + 512)
                    # )
                    jpg_img = io.BytesIO()
                    image_rgb.save(jpg_img, format="JPEG")

                    base64_img = base64.b64encode(jpg_img.getvalue()).decode('utf-8')
                    # print(f"\nConverted image to base64: {base64_img[:20]}...\n")

                    payload = {
                        "model": "bakllava",
                        "prompt": input.prompt,
                        "images": [base64_img],
                    }
                    # print(f"Sending payload to {self._url}: {payload}")

                    async with session.post(self._url, json=payload) as response:
                        # print(f"\nReceived response from {self._url}\n")
                        # Initialize an empty string to hold all response fields
                        final_output = ''
                        # Process streaming response
                        #TODO: fix this it is hacky and drops responses
                        async for chunk in response.content.iter_any():
                            chunk_str = chunk.decode('utf-8')
                            if chunk_str.strip():
                                try:
                                    chunk_json = json.loads(chunk_str)
                                    final_output += chunk_json['response']
                                    if chunk_json['done']:
                                        break
                                except json.JSONDecodeError as e:
                                    logging.error(f"Error decoding JSON: {e} {chunk}")
                                    continue
                        
                        # print(f"\nFinal output: {final_output}\n")
                        # Put the final output into the output queue
                        if len(final_output) > 5 and "unchanged" not in final_output.lower():
                            await self._output_queue.put(final_output)

                except asyncio.CancelledError:
                    print("Task cancelled")
                    break
                except Exception as e:
                    logging.error(f"Error processing frame: {e}")

    def __anext__(self) -> str:
        if self._closed and self._output_queue.empty():
            raise StopAsyncIteration

        return self._output_queue.get()

    def __aiter__(self):
        return self