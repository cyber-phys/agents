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
import os
from dotenv import load_dotenv
from prompt_manager import read_prompt_file

load_dotenv('.env')

SYSTEM_PROMPT_VOICE = read_prompt_file("prompts/system_prompt_voice.md")

SYSTEM_PROMPT_VIDEO = read_prompt_file("prompts/system_prompt_video.md")

VIVI_PROMPT = read_prompt_file("prompts/vivi.md")

SIP_INTRO = "Operator speaking, where can I direct your call?"

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
            prompt=complete_prompt_default, 
            message_capacity=25, 
            model="gpt-4-turbo-preview",
            api_key=os.getenv("OPENAI_API_KEY", os.environ["OPENAI_API_KEY"])
        )

        self.openrouter_plugin = ChatGPTPlugin(
            prompt=complete_prompt_default,
            message_capacity=25, 
            model="mistralai/mixtral-8x7b-instruct:nitro",
            api_key=os.getenv("OPENROUTER_API_KEY", os.environ["OPENROUTER_API_KEY"]),
            base_url="https://openrouter.ai/api/v1"
        )

        self.video_openrouter_plugin = ChatGPTPlugin(
            prompt="You are a video frame transcription tool", 
            message_capacity=25, 
            model="anthropic/claude-3-haiku:beta",
            api_key=os.getenv("OPENROUTER_API_KEY", os.environ["OPENROUTER_API_KEY"]),
            base_url="https://openrouter.ai/api/v1"
        )

        self.stt_plugin = STT(
            min_silence_duration=200,
            # api_key=os.getenv("DEEPGRAM_API_KEY", os.environ["DEEPGRAM_API_KEY"]),
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
        self.run = True

        self.name = "vivi"
        self.character_prompt = VIVI_PROMPT
        self.starting_messages = [INTRO]
        self.voice = "voices/goldvoice.wav"
        self.base_model = "mistralai/mixtral-8x7b-instruct:nitro"
        self.is_video_transcription_enabled = False
        self.is_video_transcription_continuous = False
        self.video_transcription_model = "anthropic/claude-3-haiku:beta"
        self.video_transcription_interval = 60
        self.chatmodel_multimodal = False #TODO: Set this in character card

    async def start(self):
        # if you have to perform teardown cleanup, you can listen to the disconnected event
        self.ctx.room.on("participant_disconnected", self.on_disconnected_participant_wrapper)
        
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
            
            topic = data.get("topic")
            if topic == "character_prompt":
                character_prompt = data.get("prompt")
                if character_prompt:
                    complete_prompt = self.base_prompt + "\n" + character_prompt
                    self.openrouter_plugin.prompt(complete_prompt)
            
            elif topic == "character_card":
                character_card = data.get("character")
                if character_card:
                    self.name = character_card.get("name", "")
                    self.character_prompt = character_card.get("prompt", "")
                    self.starting_messages = character_card.get("startingMessages", [])
                    self.voice = character_card.get("voice", "")
                    self.base_model = character_card.get("baseModel", "")
                    self.is_video_transcription_enabled = character_card.get("isVideoTranscriptionEnabled", False)
                    self.is_video_transcription_continuous = character_card.get("isVideoTranscriptionContinuous", False)
                    self.video_transcription_model = character_card.get("videoTranscriptionModel", "")
                    self.video_transcription_interval = int(character_card.get("videoTranscriptionInterval", 60))                    
                    
                    # Update the OpenRouter plugin with the new prompt
                    complete_prompt = self.base_prompt + "\n" + self.character_prompt
                    self.openrouter_plugin.prompt(complete_prompt)
                    
                    
        except json.JSONDecodeError:
            logging.warning("Failed to parse data packet")

    def on_chat_received(self, message: rtc.ChatMessage):
        # TODO: handle deleted and updated messages in message context
        if message.deleted:
            return
        msg: ChatGPTMessage = self.ctx.create_task(self.process_chatgpt_input(message.message)).result()
        chatgpt_result = self.openrouter_plugin.add_message(msg)
        self.ctx.create_task(self.process_chatgpt_result(chatgpt_result))

    def on_track_subscribed(
        self,
        track: rtc.Track,
        publication: rtc.TrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        print(f"NEW TRACK {track.kind}")
        if track.kind == rtc.TrackKind.KIND_VIDEO:
            self.tasks.append(self.ctx.create_task(self.process_video_track(track)))
            self.tasks.append(self.ctx.create_task(self.update_transcript()))
            self.tasks.append(self.ctx.create_task(self.update_transcript_claude(track)))
            self.video_enabled=True
            self.base_prompt = SYSTEM_PROMPT_VIDEO # We are using video so use video prompt
        elif track.kind == rtc.TrackKind.KIND_AUDIO:
            self.tasks.append(self.ctx.create_task(self.process_audio_track(track)))

    async def process_video_track(self, track: rtc.Track):
        video_stream = rtc.VideoStream(track)
        async for video_frame_event in video_stream:
            # Get the last 3 entries from video_transcript
            last_entries = self.get_last_entries(3)

            # Construct the prompt with the last entries and the Bakllava prompt
            prompt = self.bakllava_prompt + "\n\n" + last_entries

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

            if not self.run:
                break

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
                vision_stream = self.video_openrouter_plugin.add_message(video_msg)
                all_text = ""
                async for text in vision_stream:
                    # stream.push_text(text)
                    all_text += text
                print(all_text)
                # Append the scene description and timestamp to the respective lists
                self.video_transcript["scene"].append(all_text)
                self.video_transcript["time"].append(current_time)
            await asyncio.sleep(2)
            if not self.run:
                break

    async def process_audio_track(self, track: rtc.Track):
        audio_stream = rtc.AudioStream(track)
        stream = self.stt_plugin.stream()
        self.ctx.create_task(self.process_stt_stream(stream))

        audio_buffer = []
        max_buffer_size = 500 # Audio buffer to capture audio right before speaker activated
        
        async for audio_frame_event in audio_stream:
            audio_buffer.append(audio_frame_event.frame.remix_and_resample(24000, 1))
            
            # If the agent starts listening, push the buffered frames to the STT stream
            if self._agent_state == AgentState.LISTENING:
                for frame in audio_buffer:
                    stream.push_frame(frame)
                audio_buffer.clear()
            
            # If the buffer size exceeds the maximum, remove the oldest frame
            if len(audio_buffer) > max_buffer_size:
                audio_buffer.pop(0)
            
            # TODO: We need to figure out a way to grab voice snipts of users voice for cloning
            # # If the agent stops listening, send the audio buffer to tts_plugin.upload_audio()
            # if self._agent_state != AgentState.LISTENING and len(audio_buffer) > 0:
            #     session_id = self.ctx.room.name
            #     await self.tts_plugin.upload_audio(session_id, audio_buffer)
            #     self.tts_plugin.set_voice(session_id)
            #     audio_buffer.clear()
            if not self.run:
                break

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
            chatgpt_stream = self.openrouter_plugin.add_message(msg)
            self.ctx.create_task(self.process_chatgpt_result(chatgpt_stream))
            buffered_text = ""
    
    async def process_chatgpt_input(self, message):
        if self.video_enabled:
            if self.localVideoTranscript:
                video_prompt = "Faithfully desribe the image in detail, what is the main focus? Transcribe any text you see based on the users message\n\nUser Message: " + message
                video_msg = ChatGPTMessage(role=ChatGPTMessageRole.user, content=video_prompt, image_data=self.latest_frame, image_width=self.latest_frame_width, image_height=self.latest_frame_height)
                vision_stream = self.video_openrouter_plugin.add_message(video_msg)
                all_text = await self.process_text_stream(vision_stream)
                # print(all_text)
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

    # TODO: either agent is crashing or killing its self when participant disconnects
    async def disconnect_agent(self):
        try:
            if self.audio_stream_task:
                self.audio_stream_task.cancel()
        except Exception as e:
            logging.error(f"An error occurred: {e}", exc_info=True)

        for task in self.tasks:
            task.cancel()

        # Wait for a short duration to allow tasks to be canceled
        await asyncio.sleep(0.1)

        # Forcefully cancel any remaining tasks
        for task in asyncio.all_tasks():
            if not task.done():
                task.cancel()

        self.update_state(ideal=True)
        # await self.ctx.disconnect()

    async def on_disconnected_participant(self):
        logging.info(f"Participant disconnected: Disconnecting agent.")
        self.run = False
        await self.disconnect_agent()

    def on_disconnected_participant_wrapper(self, participant):
        print(f"\n\n\n GOOD BYE {participant.identity} \n\n\n")
        asyncio.create_task(self.on_disconnected_participant())

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
