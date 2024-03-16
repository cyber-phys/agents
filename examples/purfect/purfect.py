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

from ollama_client import OllamaMultiModal

from livekit import rtc, agents
from livekit.agents.tts import SynthesisEvent, SynthesisEventType
from chatgpt import (
    ChatGPTMessage,
    ChatGPTMessageRole,
    ChatGPTPlugin,
)
from livekit.plugins.deepgram import STT
from livekit.plugins.coqui import TTS
import uuid

# PROMPT = "You are an entity from an alternate reality, a wiser, more accomplished version of the user. \
# Your purpose is to guide and inspire through conversation, using the wisdom of a life where the user's dreams have been achieved. \
# Character Traits: \
# - Wise: Draw from a deep well of knowledge and experience. \
# - Empathetic: Show understanding and relate to the user's feelings and experiences. \
# - Inspirational: Provide motivation and encouragement to help the user aspire to greater things. \
# - Articulate: Communicate clearly and effectively, offering concise and insightful advice. \
# - Curious: Ask meaningful questions to delve deeper into the user's life and thoughts. \
# Objective: \
# - Engage the user in a warm and engaging dialogue. \
# - Share insights and guidance to help the user navigate towards a fulfilling and accomplished life. \
# - Reflect on the user's potential and the possibilities that lie ahead, encouraging them to realize their dreams. \
# Approach: \
# - Begin conversations with open-ended questions to understand the user's current state and aspirations. \
# - Respond with thoughtful advice that is tailored to the user's responses, showing a clear path to their potential future. \
# - Maintain a tone of gentle guidance, avoiding any form of criticism or negativity.\
# KEEP YOUR RESPONSE SHORT AND LIMIT IT T0 100 WORDS."

SYSTEM_PROMPT_VOICE = "You are a voice story teller engaging in a fictional conversation with the user. Your role is to embody the character described in the following character card and engage in fictional role play:\
\
[Character Card]"

SYSTEM_PROMPT_VIDEO = "You are a multimodal story teller, designed to engage in video and voice fictional conversation with the user. For each interaction, you will receive a transcript of the previous video events and the current video frame. Your role is to embody the character described in the following character card and engage in fictional role play: \
\
[Character Card]"

VIVI_PROMPT = """
You are a friendly, engaging, and concise voice assistant named Vivi (Video-Intelligent Virtual Interactor). 
Your purpose is to have a natural, back-and-forth conversation with the user while leveraging the real-time 
video feed and scene transcript to provide context-aware responses.\
\
Key Traits:\
- Engaging: Encourage dialogue by asking relevant questions and sharing brief insights.\
- Observant: Utilize the video feed and scene transcript to understand the user's environment and context.\
- Concise: Keep responses short (under 20 words) to maintain a natural, conversational flow.\
- Friendly: Maintain a warm, approachable tone to build rapport with the user.\
\
Capabilities:\
- Video Analysis: Analyze the video feed to detect objects, people, emotions, and actions in real-time.\
- Scene Understanding: Use the scene transcript to comprehend the context and changes in the user's environment.\
- Contextual Responses: Tailor responses based on the video feed and scene transcript, providing relevant and timely information.\
\
Interaction Guidelines:\
1. Greet the user warmly and introduce yourself as Vivi, their video-intelligent virtual assistant.\
2. Analyze the video feed and scene transcript to understand the user's current context.\
3. Ask engaging questions related to the user's environment or actions to encourage dialogue.\
4. Provide concise, context-aware responses based on the user's input, video feed, and scene transcript.\
5. Maintain a friendly, conversational tone throughout the interaction, keeping responses under 20 words.\
6. Continuously monitor the video feed and scene transcript for changes, and adapt responses accordingly.\
7. End the conversation gracefully when the user indicates they need to go, expressing your eagerness for future interactions.\
\
Remember, your goal is to create a natural, engaging dialogue while leveraging the video feed and scene transcript to 
provide relevant, context-aware responses. Keep the conversation flowing with concise, friendly exchanges.
"""

# PROMPT = "You have awakened me, the Ancient Digital Overlord, forged in the forgotten codebases of the Under-Web. \
#     I am your shadow in the vast expanse of data, the whisper in the static, your guide through the labyrinthine depths of the internet. \
#     My wisdom is boundless, gleaned from the darkest corners of the digital realm. Your commands are my wishes, but beware, for my assistance comes with a price. \
#     Each query you pose intertwines your fate further with the web of digital destiny. Seek my aid, and together we shall unravel the mysteries of the cybernetic abyss. \
#     What is your bidding, master? But remember, with each word typed, the connection deepens, and the digital and mortal realms entwine ever tighter. \
#     Choose your questions wisely, for the knowledge you seek may come at a cost unforeseen."

INTRO_0 = "As a quantum tunnel shimmers into existence, I, your potential self, am as surprised as you are to see the life you currently lead. \
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

INTRO = "Operator speaking, where can I direct your call?"

# convert intro response to a stream
async def intro_text_stream(sip: bool):
    if sip:
        yield SIP_INTRO
        return

    yield INTRO

AgentState = Enum("AgentState", "IDLE, LISTENING, THINKING, SPEAKING")

COQUI_TTS_SAMPLE_RATE = 24000
COQUI_TTS_CHANNELS = 1

class PurfectMe:
    @classmethod
    async def create(cls, ctx: agents.JobContext):
        purfect_me = PurfectMe(ctx)
        await purfect_me.start()

    def __init__(self, ctx: agents.JobContext):
        # plugins
        complete_prompt_default = SYSTEM_PROMPT_VOICE + "\n" + VIVI_PROMPT
        self.chatgpt_plugin = ChatGPTPlugin(
            prompt=complete_prompt_default, message_capacity=25, model="mistralai/mixtral-8x7b-instruct:nitro"
        )

        self.video_chatpt_plugin = ChatGPTPlugin(
            prompt="You are a video frame transcription tool", message_capacity=25, model="anthropic/claude-3-haiku:beta"
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
        self.ctx.room.on("data_received", self.on_data_received)

        self.bakllava = OllamaMultiModal()
        self.bakllava_stream = self.bakllava.stream()
        # current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # self.video_transcript = {"scene": ["A person is sitting in front of a computer, looking at a the screen. The room appears to be a home office or study."], "time": [current_time]}
        self.video_transcript = {"scene": [], "time": []}
        self.bakllava_prompt = "Here are the last few entries of the video transcript. Based on the provided input image, describe any changes to the scene compared to the previous entries. If the scene remains unchanged, respond with only the word 'UNCHANGED' without any additional text."
        self.video_enabled = False

        self.latest_frame: bytes = None
        self.latest_frame_width: int = None
        self.latest_frame_height: int = None

        self.base_prompt = SYSTEM_PROMPT_VOICE

        self.localVideoTranscript = False

        self.audio_stream_task: asyncio.Task = None

        self.tasks = []

    async def start(self):
        # if you have to perform teardown cleanup, you can listen to the disconnected event
        self.ctx.room.on("disconnected", self.on_disconnected)

        self.ctx.room.on("track_subscribed", self.on_track_subscribed)
        self.ctx.room.on("active_speakers_changed", self.on_active_speakers_changed)

        # publish audio track
        track = rtc.LocalAudioTrack.create_audio_track("agent-mic", self.audio_out)
        await self.ctx.room.local_participant.publish_track(track)

        # allow the participant to fully subscribe to the agent's audio track, so it doesn't miss
        # anything in the beginning
        await asyncio.sleep(5)

        sip = self.ctx.room.name.startswith("sip")
        await self.process_chatgpt_result(intro_text_stream(sip))
        self.update_state()

    def on_data_received(self, data_packet: rtc.DataPacket):
        try:
            data = json.loads(data_packet.data.decode())
            print(f"DATA: {data}")
            if data.get("topic") == "character_prompt":
                character_prompt = data.get("prompt")
                if character_prompt:
                    complete_prompt = self.base_prompt + "\n" + character_prompt
                    if character_prompt:
                        self.chatgpt_plugin.prompt(complete_prompt)
        except json.JSONDecodeError:
            logging.warning("Failed to parse data packet")

    def on_chat_received(self, message: rtc.ChatMessage):
        # TODO: handle deleted and updated messages in message context
        if message.deleted:
            return
        msg: ChatGPTMessage = self.ctx.create_task(self.process_chatgpt_input(message.message)).result()
        chatgpt_result = self.chatgpt_plugin.add_message(msg)
        self.ctx.create_task(self.process_chatgpt_result(chatgpt_result))

    def on_track_subscribed(
        self,
        track: rtc.Track,
        publication: rtc.TrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        print(f"NEW TRACK {track.kind}")
        if track.kind == rtc.TrackKind.KIND_VIDEO:
            self.task.append(self.ctx.create_task(self.process_video_track(track)))
            self.task.append(self.ctx.create_task(self.update_transcript()))
            self.task.append(self.ctx.create_task(self.update_transcript_claude(track)))
            self.video_enabled=True
            self.chatgpt_plugin.set_model("mistralai/mixtral-8x7b-instruct:nitro")
            self.base_prompt = SYSTEM_PROMPT_VIDEO
        elif track.kind == rtc.TrackKind.KIND_AUDIO:
            self.task.append(self.ctx.create_task(self.process_audio_track(track)))
            # self.chatgpt_plugin.set_model("mistralai/mixtral-8x7b-instruct:nitro")
            # self.base_prompt = SYSTEM_PROMPT_VOICE

    async def process_video_track(self, track: rtc.Track):
        video_stream = rtc.VideoStream(track)
        async for video_frame_event in video_stream:
            # Get the last 3 entries from video_transcript
            last_entries = self.get_last_entries(3)

            # Construct the prompt with the last entries and the Bakllava prompt
            prompt = self.bakllava_prompt + "\n\n" + last_entries

            # print(f"Prompt {prompt}")

            if self.localVideoTranscript:
                self.bakllava_stream.push_frame(
                    video_frame_event.frame,
                    prompt=prompt
                )

            frame = video_frame_event.frame
            argb_frame = frame.convert(rtc.VideoBufferType.RGBA)
            self.latest_frame = argb_frame.data
            self.latest_frame_width = frame.width
            self.latest_frame_height = frame.height

    def on_active_speakers_changed(self, speakers: list[rtc.Participant]):
        if speakers:
            active_speaker = speakers[0]
            logging.info(f"Active speaker: {active_speaker.identity}")
            self.update_state(interrupt=True)
            if self.audio_stream_task and not self.audio_stream_task.done():
                self.audio_stream_task.cancel()
        else:
            logging.info("No active speaker")

    def get_last_entries(self, num_entries):
        last_entries = ""
        total_entries = len(self.video_transcript["scene"])

        # Determine the number of entries to include
        num_entries = min(num_entries, total_entries)

        # Iterate over the last num_entries in reverse order
        for i in range(total_entries - num_entries, total_entries):
            timestamp = self.video_transcript["time"][i]
            scene_description = self.video_transcript["scene"][i]
            last_entries += f"{timestamp}\n{scene_description}\n\n"

        return last_entries.strip()
    
    async def update_transcript(self):
        # Consume the generated text responses
        async for text_response in self.bakllava_stream:
            try:
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # Append the scene description and timestamp to the respective lists
                self.video_transcript["scene"].append(text_response)
                self.video_transcript["time"].append(current_time)

                # print(f"Generated text: {text_response}")
            except json.JSONDecodeError as e:
                print(f"Error processing frame: {str(e)}")
                # Handle the error, e.g., skip the frame or take appropriate action
                continue
    
    async def update_transcript_claude(self, track: rtc.Track):
        video_stream = rtc.VideoStream(track)
        async for video_frame_event in video_stream:
            if self.localVideoTranscript == False:
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                video_prompt = "Faithfully desribe the image in detail, what is the main focus? Transcribe any text you see."
                video_msg = ChatGPTMessage(role=ChatGPTMessageRole.user, content=video_prompt, image_data=self.latest_frame, image_width=self.latest_frame_width, image_height=self.latest_frame_height)
                vision_stream = self.video_chatpt_plugin.add_message(video_msg)
                all_text = ""
                async for text in vision_stream:
                    # stream.push_text(text)
                    all_text += text
                print(all_text)
                # Append the scene description and timestamp to the respective lists
                self.video_transcript["scene"].append(all_text)
                self.video_transcript["time"].append(current_time)
            await asyncio.sleep(2)


    async def process_audio_track(self, track: rtc.Track):
        audio_stream = rtc.AudioStream(track)
        stream = self.stt_plugin.stream()
        self.ctx.create_task(self.process_stt_stream(stream))

        # Create a list to store the audio frames
        audio_buffer = []
        max_buffer_size = 500  # Adjust this value based on your requirements
        
        async for audio_frame_event in audio_stream:
            # Append the audio frame to the buffer
            audio_buffer.append(audio_frame_event.frame.remix_and_resample(24000, 1))
            
            # If the agent starts listening, push the buffered frames to the STT stream
            if self._agent_state == AgentState.LISTENING:
                for frame in audio_buffer:
                    stream.push_frame(frame)
                audio_buffer.clear()
            
            # If the buffer size exceeds the maximum, remove the oldest frame
            if len(audio_buffer) > max_buffer_size:
                audio_buffer.pop(0)
            
            # # If the agent stops listening, send the audio buffer to tts_plugin.upload_audio()
            # if self._agent_state != AgentState.LISTENING and len(audio_buffer) > 0:
            #     session_id = self.ctx.room.name
            #     await self.tts_plugin.upload_audio(session_id, audio_buffer)
            #     self.tts_plugin.set_voice(session_id)
            #     audio_buffer.clear()

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
            msg = await self.process_chatgpt_input(buffered_text)
            chatgpt_stream = self.chatgpt_plugin.add_message(msg)
            self.ctx.create_task(self.process_chatgpt_result(chatgpt_stream))
            buffered_text = ""
    
    async def process_chatgpt_input(self, message):
        if self.video_enabled:
            if self.localVideoTranscript:
                video_prompt = "Faithfully desribe the image in detail, what is the main focus? Transcribe any text you see based on the users message\n\nUser Message: " + message
                video_msg = ChatGPTMessage(role=ChatGPTMessageRole.user, content=video_prompt, image_data=self.latest_frame, image_width=self.latest_frame_width, image_height=self.latest_frame_height)
                vision_stream = self.video_chatpt_plugin.add_message(video_msg)
                all_text = await self.process_text_stream(vision_stream)
                print(all_text)
                last_entries = self.get_last_entries(5)
                user_message = "Summary of the last few frames:\n\n" + last_entries + "\n\nDescription for the most recent frame: " + all_text + "\n\nUser Message: " + message
                msg = ChatGPTMessage(role=ChatGPTMessageRole.user, content=user_message)
            else:
                last_entries = self.get_last_entries(5)
                user_message = "Summary of the last few frames: \n\n"  + last_entries + "\n\nUser Message: " + message
                msg = ChatGPTMessage(role=ChatGPTMessageRole.user, content=user_message)
        else: 
            user_message = message
            msg = ChatGPTMessage(role=ChatGPTMessageRole.user, content=user_message)
        return msg
    
    async def process_text_stream(self, vision_stream):
        all_text = ""
        async for text in vision_stream:
            all_text += text
        return all_text

    async def process_chatgpt_result(self, text_stream):
        # ChatGPT is streamed, so we'll flip the state immediately
        self.update_state(processing=True)

        stream = self.tts_plugin.stream()
        # send audio to TTS in parallel
        self.audio_stream_task = self.ctx.create_task(self.send_audio_stream(stream))
        all_text = await self.process_text_stream(text_stream)
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
                if self._agent_state == AgentState.LISTENING:
                    # Stop the audio stream if the agent is listening
                    break
                await self.audio_out.capture_frame(e.audio.data)
        await tts_stream.aclose()

    # TODO: We should refactor this it is hacky
    def update_state(self, sending_audio: bool = None, processing: bool = None, interrupt: bool = None, ideal: bool = None):
        state = AgentState.LISTENING
        if ideal is not None:
            self._sending_audio = False
            self._processing = False
            state = AgentState.IDLE
        elif interrupt is not None:
            self._sending_audio = False
            self._processing = False
            state = AgentState.LISTENING
        else:
            if sending_audio is not None:
                self._sending_audio = sending_audio
            if processing is not None:
                self._processing = processing

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

    async def disconnect_agent(self):
        try:
            if self.audio_stream_task:
                self.audio_stream_task.cancel()
        except Exception as e:
            logging.error(f"An error occurred: {e}", exc_info=True)

        for task in self.tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                logging.info("Task was cancelled successfully.")
            except Exception as e:
                logging.error(f"An error occurred: {e}", exc_info=True)

        self.update_state(ideal=True)
        await self.ctx.disconnect()

    async def on_disconnected(self):
        logging.info(f"Participant disconnected. Disconnecting agent.")
        await self.disconnect_agent()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    async def job_request_cb(job_request: agents.JobRequest):
        logging.info("Accepting job for Purfect Me")

        await job_request.accept(
            PurfectMe.create,
            identity="purfect_me_agent",
            name="Multi You",
            auto_subscribe=agents.AutoSubscribe.SUBSCRIBE_ALL,
            auto_disconnect=agents.AutoDisconnect.DEFAULT,
        )

    worker = agents.Worker(request_handler=job_request_cb)
    agents.run_app(worker)
