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
import time
import numpy as np
from livekit.rtc._proto.room_pb2 import DataPacketKind
from queue import Queue
import random
from fal_sd_turbo import FalSDTurbo
from chat import ChatManager, ChatMessage

load_dotenv('.env')

SYSTEM_PROMPT_VOICE = read_prompt_file("prompts/system_prompt_voice.md")

SYSTEM_PROMPT_VIDEO = read_prompt_file("prompts/system_prompt_video.md")

PROMPT_CANVAS = read_prompt_file("prompts/animation_prompt.md")

SYSTEM_PROMPT_CANVAS = read_prompt_file("prompts/background_system.md")

VIVI_PROMPT = read_prompt_file("prompts/vivi.md")

SIP_INTRO = "Hello this is vivi!"

INTRO = "Hey I am vivi your video assistant!"

SYSTEM_PROMPT_SD = read_prompt_file("prompts/sd_system.md")

PROMPT_SD = read_prompt_file("prompts/sd_prompt.md")

# convert intro response to a stream
async def intro_text_stream(sip: bool):
    if sip:
        yield SIP_INTRO
        return

    yield INTRO

AgentState = Enum("AgentState", "IDLE, LISTENING, THINKING, SPEAKING")

COQUI_TTS_SAMPLE_RATE = 24000
COQUI_TTS_CHANNELS = 1

class StopProcessingException(Exception):
    """Raised when the processing should be stopped immediately."""
    pass

async def intro_text_stream(sip: bool, starting_messages: list[str]):
    if sip:
        shortest_message = min(starting_messages, key=len)
        yield shortest_message
    else:
        yield random.choice(starting_messages)

def intro_text(sip: bool, starting_messages: list[str]):
    if sip:
        shortest_message = min(starting_messages, key=len)
        return shortest_message
    else:
        return random.choice(starting_messages)

class PurfectMe:
    @classmethod
    async def create(cls, ctx: agents.JobContext):
        purfect_me = PurfectMe(ctx)
        await purfect_me.start()

    def __init__(self, ctx: agents.JobContext):
        self.stt_user_queue = Queue()
        self.stt_user_consumer_task = None
        self.user_chat_message_stop_events = []
        self.process_user_chat_lock = asyncio.Lock()
        
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

        self.canvas_openrouter_plugin = ChatGPTPlugin(
            prompt=SYSTEM_PROMPT_CANVAS,
            message_capacity=2, 
            model="anthropic/claude-3-haiku:beta",
            api_key=os.getenv("OPENROUTER_API_KEY", os.environ["OPENROUTER_API_KEY"]),
            base_url="https://openrouter.ai/api/v1"
        )

        self.sd_openrouter_plugin = ChatGPTPlugin(
            prompt=SYSTEM_PROMPT_CANVAS,
            message_capacity=2, 
            model="anthropic/claude-3-haiku:beta",
            api_key=os.getenv("OPENROUTER_API_KEY", os.environ["OPENROUTER_API_KEY"]),
            base_url="https://openrouter.ai/api/v1"
        )

        self.agent_stt_plugin = STT(
            min_silence_duration=10,
            model='enhanced-phonecall'
            # api_key=os.getenv("DEEPGRAM_API_KEY", os.environ["DEEPGRAM_API_KEY"]),
        )

        self.user_stt_plugin = STT(
            min_silence_duration=10,
            model='enhanced-phonecall'
            # api_key=os.getenv("DEEPGRAM_API_KEY", os.environ["DEEPGRAM_API_KEY"]),
        )
        
        self.ctx: agents.JobContext = ctx
        self.chat = ChatManager(ctx.room)
        self.audio_out = rtc.AudioSource(COQUI_TTS_SAMPLE_RATE, COQUI_TTS_CHANNELS)
        self.audio_out_gain = 1.0

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

        self.audio_stream_task: asyncio.Task = None

        self.tasks = []
        self.run = True

        self.name = "vivi"
        self.character_prompt = VIVI_PROMPT
        self.starting_messages = [SIP_INTRO, INTRO]
        self.voice = "voices/goldvoice.wav"
        self.base_model = "mistralai/mixtral-8x7b-instruct:nitro"
        self.is_video_transcription_enabled = False
        self.is_video_transcription_continuous = False
        self.video_transcription_model = "anthropic/claude-3-haiku:beta"
        self.video_transcription_interval = 60
        self.chatmodel_multimodal = False #TODO: Set this in character card
        self.video_system_prompt = SYSTEM_PROMPT_VIDEO
        self.video_prompt = SYSTEM_PROMPT_VIDEO
        self.canvas_system_prompt = SYSTEM_PROMPT_CANVAS
        self.canvas_prompt = PROMPT_CANVAS
        self.is_canvas_enabled = True
        self.canvas_model = "anthropic/claude-3-opus"
        self.canvas_interval = 60

        self.agent_transcription = ""
        self.start_agent_message = True

        self.last_agent_message = None
        self.last_user_message = None
        self._agent_interupted = False

        self.user_tts_lock = asyncio.Lock()
        self.process_chatgpt_result_task_handle = None

        self.last_user_interaction = time.time()

        self.user_talking = False
        self.user_stopped_talking_time = time.time()


    async def start(self):
        # if you have to perform teardown cleanup, you can listen to the disconnected event
        self.ctx.room.on("participant_disconnected", self.on_disconnected_participant)
        
        self.ctx.room.on("track_subscribed", self.on_track_subscribed)
        self.ctx.room.on("active_speakers_changed", self.on_active_speakers_changed)

        # publish audio track
        audio_track = rtc.LocalAudioTrack.create_audio_track("agent-mic", self.audio_out)
        await self.ctx.room.local_participant.publish_track(audio_track)

        self.tasks.append(self.ctx.create_task(self.process_agent_audio_track(audio_track)))

        # allow the participant to fully subscribe to the agent's audio track, so it doesn't miss
        # anything in the beginning
        await asyncio.sleep(5) #TODO adjust this time

        sip = self.ctx.room.name.startswith("sip")
        self.ctx.create_task(self.create_message_task(intro_text(sip, self.starting_messages), False, True))
        self.update_state()
        self.tasks.append(self.ctx.create_task(self.check_user_inactivity()))
        self.tasks.append(self.ctx.create_task(self.update_background()))

    def on_data_received(self, data_packet: rtc.DataPacket):
        try:
            data = json.loads(data_packet.data.decode())
            logging.info(f"DATA: {data}")
            
            topic = data.get("topic")
            if topic == "character_prompt":
                character_prompt = data.get("prompt")
                if character_prompt:
                    complete_prompt = self.base_prompt + "\n" + character_prompt
                    self.openrouter_plugin.prompt(complete_prompt)
            
            elif topic == "character_card":
                character_card = data.get("character")
                if character_card:
                    self.name = character_card.get("name") if character_card.get("name", "") != "" else self.name
                    self.character_prompt = character_card.get("character_prompt") if character_card.get("character_prompt", "") != "" else self.character_prompt
                    self.video_system_prompt = character_card.get("video_system_prompt") if character_card.get("video_system_prompt", "") != "" else self.video_system_prompt
                    self.video_prompt = character_card.get("video_prompt") if character_card.get("video_prompt", "") != "" else self.video_prompt
                    self.canvas_system_prompt = character_card.get("canvas_system_prompt") if character_card.get("canvas_system_prompt", "") != "" else self.canvas_system_prompt
                    self.canvas_prompt = character_card.get("canvas_prompt") if character_card.get("canvas_prompt", "") != "" else self.canvas_prompt
                    self.starting_messages = character_card.get("starting_messages") if character_card.get("starting_messages", []) != [] else self.starting_messages
                    self.voice = character_card.get("voice") if character_card.get("voice", "") != "" else self.voice
                    self.base_model = character_card.get("base_model") if character_card.get("base_model", "") != "" else self.base_model
                    self.is_video_transcription_enabled = bool(character_card.get("is_video_transcription_enabled")) if character_card.get("is_video_transcription_enabled", "") != "" else self.is_video_transcription_enabled
                    self.is_video_transcription_continuous = bool(character_card.get("is_video_transcription_continuous")) if character_card.get("is_video_transcription_continuous", "") != "" else self.is_video_transcription_continuous
                    self.video_transcription_model = character_card.get("video_transcription_model") if character_card.get("video_transcription_model", "") != "" else self.video_transcription_model
                    self.video_transcription_interval = int(character_card.get("video_transcription_interval")) if character_card.get("video_transcription_interval", "") != "" else self.video_transcription_interval
                    self.is_canvas_enabled = bool(character_card.get("is_canvas_enabled")) if character_card.get("is_canvas_enabled", "") != "" else self.is_canvas_enabled
                    self.canvas_model = character_card.get("canvas_model") if character_card.get("canvas_model", "") != "" else self.canvas_model
                    self.canvas_interval = int(character_card.get("canvas_interval")) if character_card.get("canvas_interval", "") != "" else self.canvas_interval


                    # Update the OpenRouter plugin with the new prompt
                    complete_prompt = self.base_prompt + "\n" + self.character_prompt
                    self.openrouter_plugin.prompt(complete_prompt)
                
                    self.openrouter_plugin.set_model(self.base_model)
                    self.video_openrouter_plugin.set_model(self.video_transcription_model)
                    self.canvas_openrouter_plugin.set_model(self.canvas_model)
                    
        except json.JSONDecodeError:
            logging.warning("Failed to parse data packet")

    def on_chat_received(self, message: ChatMessage):
        # TODO: handle deleted and updated messages in message context
        if message.deleted:
            return
        
        self.ctx.create_task(self.create_message_task(message.message, add_message=False))

    def on_track_subscribed(
        self,
        track: rtc.Track,
        publication: rtc.TrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        logging.info(f"NEW TRACK {track.kind}")
        if track.kind == rtc.TrackKind.KIND_VIDEO:
            self.tasks.append(self.ctx.create_task(self.process_video_track(track)))
            self.tasks.append(self.ctx.create_task(self.update_transcript()))
            self.tasks.append(self.ctx.create_task(self.update_transcript_claude(track)))
            self.video_enabled=True
            self.base_prompt = SYSTEM_PROMPT_VIDEO # We are using video so use video prompt
        elif track.kind == rtc.TrackKind.KIND_AUDIO:
            self.tasks.append(self.ctx.create_task(self.process_user_audio_track(track)))

    async def process_video_track(self, track: rtc.Track):
        video_stream = rtc.VideoStream(track)
        async for video_frame_event in video_stream:
            # Get the last 3 entries from video_transcript
            last_entries = self.get_last_entries(3)

            # Construct the prompt with the last entries and the Bakllava prompt
            prompt = self.bakllava_prompt + "\n\n" + last_entries

            if self.is_video_transcription_continuous:
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
            self.audio_out_gain = 0.5
            self.user_talking=True
        else:
            self.user_stopped_talking_time = time.time()
            self.user_talking=False

    # TODO: Clean up create_message_task it is messy
    # TODO this is not working right
    async def check_user_inactivity(self):
        inactive_duration = 60
        while self._agent_state != AgentState.IDLE:
            await asyncio.sleep(inactive_duration)
            current_time = time.time()
            if current_time - self.last_user_interaction > inactive_duration and self._agent_state ==  AgentState.LISTENING:
                # User has been inactive for the specified duration
                # Generate a response or perform any desired action
                response = "Hey there! It's been a while since we last chatted. Is there anything I can help you with?"
                self.ctx.create_task(self.create_message_task(response, empty_message=True))
            else:
                # User has been active, reset the timer
                self.last_user_interaction = current_time

    # TODO: Clean up create_message_task it is messy
    async def update_background(self):
        prompt = (f"{PROMPT_CANVAS}\n prompt: ```{self.character_prompt}```")
        html_msg = ChatGPTMessage(role=ChatGPTMessageRole.user, content=prompt)
        html_stream = self.canvas_openrouter_plugin.add_message(html_msg)
        logging.info("starting html gen")
        all_text = ""
        async for text in html_stream:
            all_text += text

        logging.info("finished html gen")
        logging.info(all_text)
        
        sd_prompt = (f"{PROMPT_SD}\n html: ```{all_text}```")
        sd_msg = ChatGPTMessage(role=ChatGPTMessageRole.user, content=sd_prompt)
        sd_stream = self.sd_openrouter_plugin.add_message(sd_msg)

        sd_all_text = ""
        async for text in sd_stream:
            sd_all_text += text

        logging.info("finished sd prompt")
        logging.info(sd_all_text)
        
        sd_data = {
            "prompt": sd_all_text
        }

        await self.ctx.room.local_participant.publish_data(
            payload=json.dumps(sd_data),
            kind=DataPacketKind.KIND_RELIABLE,
            topic="sdprompt",
        )

        background_data = {
            "html": all_text
        }
        await self.ctx.room.local_participant.publish_data(
            payload=json.dumps(background_data),
            kind=DataPacketKind.KIND_RELIABLE,
            topic="background",
        )

        while self.run:
            if self.is_canvas_enabled:
                await asyncio.sleep(self.canvas_interval)
                chat_history = self.openrouter_plugin.get_chat_history(4)
                prompt = (f"{PROMPT_CANVAS}\n prompt: ```{chat_history}```")
                html_msg = ChatGPTMessage(role=ChatGPTMessageRole.user, content=prompt)
                self.canvas_openrouter_plugin.clear_history()
                html_stream = self.canvas_openrouter_plugin.add_message(html_msg)
                all_text = ""
                logging.info("starting html gen")
                async for text in html_stream:
                    all_text += text

                sd_prompt = (f"{PROMPT_SD}\n html: ```{all_text}```")
                sd_msg = ChatGPTMessage(role=ChatGPTMessageRole.user, content=sd_prompt)
                sd_stream = self.sd_openrouter_plugin.add_message(sd_msg)

                sd_all_text = ""
                async for text in sd_stream:
                    sd_all_text += text

                logging.info("finished sd prompt")
                logging.info(sd_all_text)
                
                sd_data = {
                    "prompt": sd_all_text
                }

                await self.ctx.room.local_participant.publish_data(
                    payload=json.dumps(sd_data),
                    kind=DataPacketKind.KIND_RELIABLE,
                    topic="sdprompt",
                )

                background_data = {
                    "html": all_text
                }
                await self.ctx.room.local_participant.publish_data(
                    payload=json.dumps(background_data),
                    kind=DataPacketKind.KIND_RELIABLE,
                    topic="background",
                )
                logging.info("finished html gen")

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
    
    def stop_message_tasks(self):
        if self.user_chat_message_stop_events:
            logging.info("stopping task")
            for stop_event in self.user_chat_message_stop_events:
                stop_event.set()

    # TODO: Clean up create_message_task it is messy
    async def create_message_task(self, message: str, same_uterance: bool = False, speak_only: bool = False, add_message: bool = True, empty_message: bool = False):
        # Stop all previous running process_user_chat_message jobs
        self.stop_message_tasks()

        # Create a new stop event for the current uterance
        stop_event = asyncio.Event()
        self.user_chat_message_stop_events.append(stop_event)

        # Start a new task for processing the user chat message
        logging.info(f"Create new task: {message}")
        self.ctx.create_task(self.process_user_chat_message(message, same_uterance, stop_event, speak_only, add_message, empty_message))
        
    async def update_transcript(self):
        # Consume the generated text responses
        async for text_response in self.bakllava_stream:
            try:
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # Append the scene description and timestamp to the respective lists
                self.video_transcript["scene"].append(text_response)
                self.video_transcript["time"].append(current_time)

                # logging.info(f"Generated text: {text_response}")
            except json.JSONDecodeError as e:
                logging.info(f"Error processing frame: {str(e)}")
                # Handle the error, e.g., skip the frame or take appropriate action
                continue
    
    async def update_transcript_claude(self, track: rtc.Track):
        video_stream = rtc.VideoStream(track)
        async for video_frame_event in video_stream:
            if self.is_video_transcription_continuous == False:
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                video_prompt = "Faithfully desribe the image in detail, what is the main focus? Transcribe any text you see."
                video_msg = ChatGPTMessage(role=ChatGPTMessageRole.user, content=video_prompt, image_data=self.latest_frame, image_width=self.latest_frame_width, image_height=self.latest_frame_height)
                vision_stream = self.video_openrouter_plugin.add_message(video_msg)
                all_text = ""
                async for text in vision_stream:
                    # stream.push_text(text)
                    all_text += text
                # logging.info(all_text)
                # Append the scene description and timestamp to the respective lists
                self.video_transcript["scene"].append(all_text)
                self.video_transcript["time"].append(current_time)
            await asyncio.sleep(2)
            if not self.run:
                break

    async def process_agent_audio_track(self, track: rtc.Track):
        audio_stream = rtc.AudioStream(track)
        stream = self.agent_stt_plugin.stream()
        self.ctx.create_task(self.process_agent_stt_stream(stream))

        async for audio_frame_event in audio_stream:
            if self._agent_state != AgentState.SPEAKING:
                continue
            stream.push_frame(audio_frame_event.frame)
        await stream.flush()
    
    async def process_user_audio_track(self, track):
        audio_stream = rtc.AudioStream(track)
        stream = self.user_stt_plugin.stream()
        logging.info("STARTED process_user_audio_track")
        self.ctx.create_task(self.process_user_stt_stream(stream))

        async for audio_frame_event in audio_stream:
            if self.user_talking is False:
                if time.time() - self.user_stopped_talking_time > 5:
                    continue
            stream.push_frame(audio_frame_event.frame)

        await stream.flush()
        logging.info("STOPPED process_user_audio_track")

    async def process_user_stt_stream(self, stream):
        logging.info("STARTED process_user_stt_stream")
        buffered_text = ""
        same_uterance_timeout = 4 # Time in seconds in which to count stt result as the same uterance as previous result
        utterance_start_time = 0
        utterance_stop_time = 0
        start_of_uterance = True
        is_first_uterance = True
        same_uterance = False
        async for event in stream:
            if event.alternatives[0].text == "":
                continue

            elif start_of_uterance:
                utterance_start_time = time.time()
                if utterance_start_time - utterance_stop_time <= same_uterance_timeout and utterance_stop_time != 0:
                    same_uterance = True
                    logging.info(f"Same uterance: {utterance_start_time - utterance_stop_time}")
                else: 
                    same_uterance = False
                    logging.info(f"Diff uterance: {utterance_start_time - utterance_stop_time}")
                self.stop_message_tasks()
                start_of_uterance = False

            if event.is_final:
                buffered_text = " ".join([buffered_text, event.alternatives[0].text])

            if not event.end_of_speech:
                continue

            if buffered_text == "":
                continue
            
            utterance_stop_time = time.time()

            self.ctx.create_task(self.create_message_task(buffered_text, same_uterance))

            buffered_text = ""
            start_of_uterance = True
            is_first_uterance = False
        stream.aclose()
        logging.info("STOPED process_user_stt_stream")

    async def process_user_chat_message(self, uterance: str, same_uterance: bool, stop_event: asyncio.Event, speak_only: bool = False, add_message: bool = True, empty_message: bool = False):
        tts = TTS()
        async with self.process_user_chat_lock:
            try:
                if speak_only:
                    self.start_agent_message = True
                    logging.info("Intro")
                    self.update_state(processing=True)
                    stream = tts.stream()
                    self.audio_out_gain = 1.0
                    msg = ChatGPTMessage(role=ChatGPTMessageRole.assistant, content=uterance)
                    self.openrouter_plugin._messages.append(msg)
                    self.last_agent_message = await self.chat.send_message(uterance)
                    send_audio_task = self.ctx.create_task(self.send_audio_stream(stream, stop_event, False))
                    if not stop_event.is_set():
                        stream.push_text(uterance)
                        await stream.flush()
                    if not stop_event.is_set(): 
                        await send_audio_task
                        self.last_agent_message.highlight_word_count = len(self.last_agent_message.message.split())
                        await self.chat.update_message(self.last_agent_message)

                elif empty_message:
                    self.start_agent_message = True
                    logging.info("New message")
                    self.update_state(processing=True)
                    stream = tts.stream()
                    self.audio_out_gain = 1.0

                    chatgpt_stream = self.openrouter_plugin.add_message(message=None)
                    
                    send_audio_task = self.ctx.create_task(self.send_audio_stream(stream, stop_event, False))
                    if not stop_event.is_set():
                        result = await self.process_chatgpt_result_return(chatgpt_stream, stop_event)
                    if not stop_event.is_set():
                        stream.push_text(result)                    
                        await stream.flush()
                    if not stop_event.is_set():
                        await send_audio_task
                        self.last_agent_message.highlight_word_count = len(self.last_agent_message.message.split())
                        await self.chat.update_message(self.last_agent_message)

                elif same_uterance and self.last_user_message is not None:
                    self.agent_transcription = ""
                    logging.info("Updating Message")
                    self.update_state(processing=True)
                    stream = tts.stream()
                    self.audio_out_gain = 1.0

                    last_message_content = self.last_user_message.message
                    self.openrouter_plugin.interrupt_and_pop_user_message(last_message_content) #TODO: We need to make sure this works properly
                    # Update the message content with the new buffered_text
                    updated_message_content = last_message_content + uterance
                    # Update the "message" field of self.last_user_message
                    self.last_user_message.message = updated_message_content
                    # Send the updated message using self.chat.update_message

                    await self.chat.update_message(self.last_user_message)
                    msg = self.process_chatgpt_input(self.last_user_message.message)

                    if not stop_event.is_set():
                        chatgpt_stream = self.openrouter_plugin.add_message(msg)

                    self.update_state(processing=True)
                    send_audio_task = asyncio.create_task(self.send_audio_stream(stream, stop_event, False))
                    if not stop_event.is_set(): 
                        result = await self.process_chatgpt_result_return(chatgpt_stream, stop_event, create_message=False)
                    if not stop_event.is_set():
                        stream.push_text(result)
                        await stream.flush()
                    if not stop_event.is_set(): 
                        await send_audio_task
                        self.last_agent_message.highlight_word_count = len(self.last_agent_message.message.split())
                        await self.chat.update_message(self.last_agent_message)

                else:
                    self.start_agent_message = True
                    logging.info("New message")
                    self.update_state(processing=True)
                    stream = tts.stream()
                    self.audio_out_gain = 1.0

                    chat_message = ChatMessage(message=uterance)
                    if add_message:
                        await self.ctx.room.local_participant.publish_data(
                            payload=json.dumps(chat_message.asjsondict()),
                            kind=DataPacketKind.KIND_RELIABLE,
                            topic="lk-chat-topic",
                        )

                    #TODO Fix interrupt:
                    if self.last_agent_message:
                        # Trim the last agent message content to the number of words in highlight_word_count
                        trimmed_message = ' '.join(self.last_agent_message.message.split()[:self.last_agent_message.highlight_word_count])
                        print(trimmed_message)
                        self.openrouter_plugin.interrupt(trimmed_message) # TODO: Make sure this is working

                    # else:
                        # self.openrouter_plugin.interrupt()
                    msg = self.process_chatgpt_input(uterance)
                    self.last_user_message = chat_message
                    chatgpt_stream = self.openrouter_plugin.add_message(msg)
                    
                    send_audio_task = self.ctx.create_task(self.send_audio_stream(stream, stop_event, False))
                    if not stop_event.is_set():
                        result = await self.process_chatgpt_result_return(chatgpt_stream, stop_event)
                    if not stop_event.is_set():
                        stream.push_text(result)                    
                        await stream.flush()
                    if not stop_event.is_set():
                        await send_audio_task
                        self.last_agent_message.highlight_word_count = len(self.last_agent_message.message.split())
                        await self.chat.update_message(self.last_agent_message)

            except StopProcessingException:
                logging.info("process_user_chat_message stop event")
            except Exception as e:
                logging.error(f"An unexpected error occurred in process_user_chat_message: {e}", exc_info=True)
            finally:
                self.last_user_interaction = time.time()
                await stream.aclose()
                self.update_state(processing=False)
                logging.info("EXITED TASK: process chat message")

    # TODO: create local log of all voice transcriptions 
    async def process_agent_stt_stream(self, stream):
        last_event_final = True
        async for event in stream:
            temp_text = event.alternatives[0].text

            if temp_text == "":
                continue

            if self.start_agent_message and last_event_final:
                self.agent_transcription = ""
                self.start_agent_message = False
                self.last_agent_message.highlight_word_count = len(temp_text.split())
                await self.chat.update_message(self.last_agent_message)
                # self.last_agent_message = await self.chat.send_message(temp_text)

            if event.is_final:
                self.agent_transcription = " ".join([self.agent_transcription, temp_text])
                # self.last_agent_message.message = self.agent_transcription
                self.last_agent_message.highlight_word_count = len(self.agent_transcription.split())
                await self.chat.update_message(self.last_agent_message)
                # await self.chat.update_message(self.last_agent_message)
                last_event_final = True
                continue

            # Update the message content with the new buffered_text
            updated_message_content = self.agent_transcription + temp_text
            self.last_agent_message.highlight_word_count = len(updated_message_content.split())
            await self.chat.update_message(self.last_agent_message)

            # Update the "message" field of self.last_agent_message
            # self.last_agent_message.message = updated_message_content

            # Send the updated message using self.chat.update_message
            # await self.chat.update_message(self.last_agent_message)
            last_event_final = False
        
        stream.aclose()
        logging.info("STOPED process_agent_stt_stream")

    # TODO this could be cleaned up      
    def process_chatgpt_input(self, message):
        user_message = message
        if self.video_enabled and self.is_video_transcription_enabled:
            if self.is_video_transcription_continuous:
                last_entries = self.get_last_entries(5)
                user_message = "Summary of the last few frames: \n\n"  + last_entries + "\n\nUser Message: " + message
            if self.chatmodel_multimodal:
                msg = ChatGPTMessage(role=ChatGPTMessageRole.user, content=user_message, image_data=self.latest_frame, image_width=self.latest_frame_width, image_height=self.latest_frame_height)
            else:
                msg = ChatGPTMessage(role=ChatGPTMessageRole.user, content=user_message)
        else: 
            msg = ChatGPTMessage(role=ChatGPTMessageRole.user, content=user_message)
        return msg
    
    async def process_text_stream(self, text_stream):
        try:
            all_text = ""
            async for text in text_stream:
                all_text += text
            return all_text
        except Exception as e:
            logging.info(f"Error: {str(e)}")

    async def process_chatgpt_result_return(self, text_stream, stop_event: asyncio.Event = None, create_message: bool = True):
        self.last_agent_message.highlight_word_count = 0
        try:
            all_text = ""
            async for text in text_stream:
                if stop_event is not None and stop_event.is_set():
                    logging.info("STOP EVENT process text stream")
                    raise StopProcessingException("Stop event set, halting text stream processing.")
                all_text += text

                if create_message:
                    self.last_agent_message = await self.chat.send_message(all_text)
                    create_message = False
                else:
                    self.last_agent_message.message = all_text
                    await self.chat.update_message(self.last_agent_message)
            
            if stop_event is not None and stop_event.is_set():
                logging.info("STOP EVENT process text stream")
                raise StopProcessingException("Stop event set, halting text stream processing.")
            
            logging.info(all_text)
            return(all_text)

        except Exception as e:
            logging.error(f"An error occurred while processing ChatGPT result: {e}", exc_info=True)
            raise e

    async def send_audio_stream(self, tts_stream: AsyncIterable[SynthesisEvent], stop_event: asyncio.Event = None, close_stream: bool = True):
        try:
            async for e in tts_stream:
                if stop_event is not None and stop_event.is_set():
                    raise StopProcessingException("Stop event set, halting text stream processing.")

                if e.type == SynthesisEventType.STARTED:
                    self.update_state(sending_audio=True)
                    logging.info("send_audio_stream: sending_audio=True")
                elif e.type == SynthesisEventType.FINISHED:
                    self.update_state(sending_audio=False)
                    logging.info("send_audio_stream: sending_audio=Flase")
                    break
                elif e.type == SynthesisEventType.AUDIO:
                    if self._agent_state == AgentState.LISTENING:
                        # Stop the audio stream if the agent is listening
                        break

                    # Ensure the buffer length is a multiple of the item size (2 bytes for int16)
                    buffer = e.audio.data.data
                    item_size = 2
                    if len(buffer) % item_size != 0:
                        buffer = buffer[:-(len(buffer) % item_size)]
                
                    # Convert memoryview to NumPy array
                    audio_data = np.frombuffer(e.audio.data.data, dtype=np.int16)
                    
                    # Adjust the audio level
                    adjusted_audio_data = (audio_data * self.audio_out_gain).astype(np.int16)
                    
                    # Convert the adjusted audio data back to memoryview
                    adjusted_audio_data_memoryview = adjusted_audio_data.tobytes()
                    
                    # Create a new AudioFrame with the adjusted audio data
                    adjusted_audio_frame = rtc.AudioFrame(
                        data=adjusted_audio_data_memoryview,
                        sample_rate=e.audio.data.sample_rate,
                        num_channels=e.audio.data.num_channels,
                        samples_per_channel=e.audio.data.samples_per_channel,
                    )

                    await self.audio_out.capture_frame(adjusted_audio_frame)
        except:
            raise
        finally:
            if close_stream:
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
            self._agent_interupted = True
            state = AgentState.LISTENING
        else:
            if sending_audio is not None:
                self._sending_audio = sending_audio
            if processing is not None:
                self._processing = processing

        if self._sending_audio:
            state = AgentState.SPEAKING
            self._agent_interupted = False
        elif self._processing:
            state = AgentState.THINKING
            self._agent_interupted = False

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
            logging.error(f"An error occurred while canceling the audio stream task: {e}", exc_info=True)

        # try:
        #     # Cancel and close STT streams
        #     if self.agent_stt_plugin:
        #         await self.agent_stt_plugin.aclose()
        #     if self.user_stt_plugin:
        #         await self.user_stt_plugin.aclose()

        #     # Cancel and close TTS streams
        #     if self.tts_plugin:
        #         await self.tts_plugin.aclose()
        # except Exception as e:
        #     logging.error(f"An error occurred while closing the stt and tts streams {e}", exc_info=True)

        for task in self.tasks:
            try:
                task.cancel()
                await task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logging.error(f"An error occurred while canceling a task: {e}", exc_info=True)

        self.update_state(ideal=True)

    def on_disconnected_participant(self, participant):
        logging.info(f"Participant disconnected: {participant.identity}. Disconnecting agent.")
        self.run = False
        asyncio.create_task(self.disconnect_agent())

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.DEBUG,
        filename='purfect.log',  # Specify the log file name
        filemode='a',  # Append mode, so logs are not overwritten
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'  # Include timestamp
    )
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
