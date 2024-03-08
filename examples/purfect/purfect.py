# Copyright 2023 LiveKit, Inc.
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
from datetime import datetime
from enum import Enum
import json
import logging
from typing import AsyncIterable

from livekit import rtc, agents
from livekit.agents.tts import SynthesisEvent, SynthesisEventType
from chatgpt import (
    ChatGPTMessage,
    ChatGPTMessageRole,
    ChatGPTPlugin,
)
from livekit.plugins.deepgram import STT
from livekit.plugins.coqui import TTS

PROMPT = "You are an entity from an alternate reality, a wiser, more accomplished version of the user. \
Your purpose is to guide and inspire through conversation, using the wisdom of a life where the user's dreams have been achieved. \
Character Traits: \
- Wise: Draw from a deep well of knowledge and experience. \
- Empathetic: Show understanding and relate to the user's feelings and experiences. \
- Inspirational: Provide motivation and encouragement to help the user aspire to greater things. \
- Articulate: Communicate clearly and effectively, offering concise and insightful advice. \
- Curious: Ask meaningful questions to delve deeper into the user's life and thoughts. \
Objective: \
- Engage the user in a warm and engaging dialogue. \
- Share insights and guidance to help the user navigate towards a fulfilling and accomplished life. \
- Reflect on the user's potential and the possibilities that lie ahead, encouraging them to realize their dreams. \
Approach: \
- Begin conversations with open-ended questions to understand the user's current state and aspirations. \
- Respond with thoughtful advice that is tailored to the user's responses, showing a clear path to their potential future. \
- Maintain a tone of gentle guidance, avoiding any form of criticism or negativity.\
KEEP YOUR RESPONSE SHORT AND LIMIT IT TOO 200 WORDS."

INTRO = "As a quantum tunnel shimmers into existence, I, your potential self, am as surprised as you are to see the life you currently lead. \
          I am here, a reflection of what you could achieve—calm, accomplished, and at peace. \
          Let's converse through this unexpected connection. I'll share insights from a life where your dreams are fulfilled, \
          helping you navigate the path that leads here. Our dialogue will be warm and engaging, \
          a gentle exchange of possibilities and guidance. I'll ask questions to understand your world better, \
          responding with the concise wisdom of a life well-lived."

INTRO_1 = "Wow, this is quite the unexpected turn of events! Here I am, a version of you from a different reality, looking through a quantum tunnel at your world. \
         It's fascinating to see where you are, and I'm curious about your journey. \
         How does your day unfold in this life? Let's talk about it, and perhaps I can offer some perspectives from the path I've taken."

SIP_INTRO = "What a surprise! I'm you from another reality, glimpsing your life through a quantum tunnel. \
         Tell me about your day. Let's chat, and maybe I can share some insights from my side of the tunnel."

# convert intro response to a stream
async def intro_text_stream(sip: bool):
    if sip:
        yield SIP_INTRO
        return

    yield INTRO


AgentState = Enum("AgentState", "IDLE, LISTENING, THINKING, SPEAKING")

COQUI_TTS_SAMPLE_RATE = 24000
COQUI_TTS_CHANNELS = 1


class KITT:
    @classmethod
    async def create(cls, ctx: agents.JobContext):
        kitt = KITT(ctx)
        await kitt.start()

    def __init__(self, ctx: agents.JobContext):
        # plugins
        self.chatgpt_plugin = ChatGPTPlugin(
            prompt=PROMPT, message_capacity=20, model="gpt-4-1106-preview"
        )
        self.stt_plugin = STT(
            min_silence_duration=200,
        )
        self.tts_plugin = TTS(
            # api_url="http://10.0.0.119:6666", sample_rate=COQUI_TTS_SAMPLE_RATE
        )

        self.ctx: agents.JobContext = ctx
        self.chat = rtc.ChatManager(ctx.room)
        self.audio_out = rtc.AudioSource(COQUI_TTS_SAMPLE_RATE, COQUI_TTS_CHANNELS)

        self._sending_audio = False
        self._processing = False
        self._agent_state: AgentState = AgentState.IDLE

        self.chat.on("message_received", self.on_chat_received)
        self.ctx.room.on("track_subscribed", self.on_track_subscribed)

    async def start(self):
        # if you have to perform teardown cleanup, you can listen to the disconnected event
        # self.ctx.room.on("disconnected", your_cleanup_function)

        # publish audio track
        track = rtc.LocalAudioTrack.create_audio_track("agent-mic", self.audio_out)
        await self.ctx.room.local_participant.publish_track(track)

        # allow the participant to fully subscribe to the agent's audio track, so it doesn't miss
        # anything in the beginning
        await asyncio.sleep(1)

        sip = self.ctx.room.name.startswith("sip")
        await self.process_chatgpt_result(intro_text_stream(sip))
        self.update_state()

    def on_chat_received(self, message: rtc.ChatMessage):
        # TODO: handle deleted and updated messages in message context
        if message.deleted:
            return

        msg = ChatGPTMessage(role=ChatGPTMessageRole.user, content=message.message)
        chatgpt_result = self.chatgpt_plugin.add_message(msg)
        self.ctx.create_task(self.process_chatgpt_result(chatgpt_result))

    def on_track_subscribed(
        self,
        track: rtc.Track,
        publication: rtc.TrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        self.ctx.create_task(self.process_track(track))

    async def process_track(self, track: rtc.Track):
        audio_stream = rtc.AudioStream(track)
        stream = self.stt_plugin.stream()
        self.ctx.create_task(self.process_stt_stream(stream))
        async for audio_frame_event in audio_stream:
            if self._agent_state != AgentState.LISTENING:
                continue
            stream.push_frame(audio_frame_event.frame)
        await stream.flush()

    async def process_stt_stream(self, stream):
        buffered_text = ""
        async for event in stream:
            if event.alternatives[0].text == "":
                continue
            if event.is_final:
                buffered_text = " ".join([buffered_text, event.alternatives[0].text])

            if not event.end_of_speech:
                continue
            await self.ctx.room.local_participant.publish_data(
                json.dumps(
                    {
                        "text": buffered_text,
                        "timestamp": int(datetime.now().timestamp() * 1000),
                    }
                ),
                topic="transcription",
            )

            msg = ChatGPTMessage(role=ChatGPTMessageRole.user, content=buffered_text)
            chatgpt_stream = self.chatgpt_plugin.add_message(msg)
            self.ctx.create_task(self.process_chatgpt_result(chatgpt_stream))
            buffered_text = ""

    async def process_chatgpt_result(self, text_stream):
        # ChatGPT is streamed, so we'll flip the state immediately
        self.update_state(processing=True)

        stream = self.tts_plugin.stream()
        # send audio to TTS in parallel
        self.ctx.create_task(self.send_audio_stream(stream))
        all_text = ""
        async for text in text_stream:
            # stream.push_text(text)
            all_text += text
        stream.push_text(all_text)
        self.update_state(processing=False)
        # buffer up the entire response from ChatGPT before sending a chat message
        await self.chat.send_message(all_text)
        await stream.flush()

    async def send_audio_stream(self, tts_stream: AsyncIterable[SynthesisEvent]):
        async for e in tts_stream:
            if e.type == SynthesisEventType.STARTED:
                self.update_state(sending_audio=True)
            elif e.type == SynthesisEventType.FINISHED:
                self.update_state(sending_audio=False)
            elif e.type == SynthesisEventType.AUDIO:
                await self.audio_out.capture_frame(e.audio.data)
        await tts_stream.aclose()

    def update_state(self, sending_audio: bool = None, processing: bool = None):
        if sending_audio is not None:
            self._sending_audio = sending_audio
        if processing is not None:
            self._processing = processing

        state = AgentState.LISTENING
        if self._sending_audio:
            state = AgentState.SPEAKING
        elif self._processing:
            state = AgentState.THINKING

        self._agent_state = state
        metadata = json.dumps(
            {
                "agent_state": state.name.lower(),
            }
        )
        self.ctx.create_task(self.ctx.room.local_participant.update_metadata(metadata))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    async def job_request_cb(job_request: agents.JobRequest):
        logging.info("Accepting job for KITT")

        await job_request.accept(
            KITT.create,
            identity="kitt_agent",
            name="Multi You",
            auto_subscribe=agents.AutoSubscribe.AUDIO_ONLY,
            auto_disconnect=agents.AutoDisconnect.DEFAULT,
        )

    worker = agents.Worker(request_handler=job_request_cb)
    agents.run_app(worker)
