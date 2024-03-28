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
from asyncore import loop
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
import threading
from queue import Queue
import random
from chat import ChatMessage, ChatManager

load_dotenv('.env')

SYSTEM_PROMPT_VOICE = read_prompt_file("prompts/system_prompt_voice.md")

SYSTEM_PROMPT_VIDEO = read_prompt_file("prompts/system_prompt_video.md")

VIVI_PROMPT = read_prompt_file("prompts/vivi.md")

SIP_INTRO = "Hello this is vivi!"

INTRO = "Hey I am vivi your video assistant!"

# convert intro response to a stream
async def intro_text_stream(sip: bool):
    if sip:
        yield SIP_INTRO
        return

    yield INTRO

#TODO we need to fix agent states
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
        self.process_user_chat_lock = threading.Lock()

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

        self.agent_stt_plugin = STT(
            min_silence_duration=200,
            # model='enhanced'
            # api_key=os.getenv("DEEPGRAM_API_KEY", os.environ["DEEPGRAM_API_KEY"]),
        )

        self.user_stt_plugin = STT(
            min_silence_duration=200,
            # model='enhanced'
            # api_key=os.getenv("DEEPGRAM_API_KEY", os.environ["DEEPGRAM_API_KEY"]),
        )

        self.tts_plugin = TTS(
            # api_url="http://10.0.0.119:6666", sample_rate=COQUI_TTS_SAMPLE_RATE
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

        self.localVideoTranscript = False

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

        self.agent_transcription = ""
        self.start_of_message = True

        self.last_agent_message = None
        self.last_user_message = None
        self._agent_interupted = False

        self.user_tts_lock = asyncio.Lock()
        self.process_chatgpt_result_task_handle = None
        self.user_tts_thread = None
        self.shared_event_loop = asyncio.new_event_loop()

    async def start(self):
        # if you have to perform teardown cleanup, you can listen to the disconnected event
        self.ctx.room.on("participant_disconnected", self.on_disconnected_participant)
        
        self.ctx.room.on("track_subscribed", self.on_track_subscribed)
        self.ctx.room.on("active_speakers_changed", self.on_active_speakers_changed)

        # publish audio track
        track = rtc.LocalAudioTrack.create_audio_track("agent-mic", self.audio_out)
        await self.ctx.room.local_participant.publish_track(track)

        self.tasks.append(self.ctx.create_task(self.process_agent_audio_track(track)))

        # allow the participant to fully subscribe to the agent's audio track, so it doesn't miss
        # anything in the beginning
        await asyncio.sleep(5) #TODO adjust this time

        #TODO we should block listening to user tts until we finish
        sip = self.ctx.room.name.startswith("sip")
        # await self.process_chatgpt_result(intro_text_stream(sip, self.starting_messages))
        self.create_message_task(intro_text(sip, self.starting_messages), False, True)
        self.update_state()
        if self.user_tts_thread:
            self.user_tts_thread.start()
        else: logging.info("TTS has not been started")

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
                    self.name = character_card.get("name", "")
                    self.character_prompt = character_card.get("prompt", "")
                    self.starting_messages = character_card.get("startingMessages", [])
                    self.voice = character_card.get("voice", "")
                    self.base_model = character_card.get("baseModel", "") #TODO check to make sure we are using this
                    self.is_video_transcription_enabled = character_card.get("isVideoTranscriptionEnabled", False) #TODO: make this control routing to video model
                    self.is_video_transcription_continuous = character_card.get("isVideoTranscriptionContinuous", False) #TODO: make this control video transcriptions
                    self.video_transcription_model = character_card.get("videoTranscriptionModel", "") #TODO use this param
                    self.video_transcription_interval = int(character_card.get("videoTranscriptionInterval", 60))                    
                    
                    # Update the OpenRouter plugin with the new prompt
                    complete_prompt = self.base_prompt + "\n" + self.character_prompt
                    self.openrouter_plugin.prompt(complete_prompt)
                    
                    
        except json.JSONDecodeError:
            logging.warning("Failed to parse data packet")

    def on_chat_received(self, message: ChatMessage):
        # TODO: handle deleted and updated messages in message context
        if message.deleted:
            return
                
        self.create_message_task(message.message)

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
            self.user_tts_thread = threading.Thread(target=self.process_user_audio_track, args=(track,))

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

    # #TODO We should wait for tts to finish
    # def interupt_agent(self):
    #     if self._agent_state == AgentState.SPEAKING:
    #         logging.info(f"\n\n{self.agent_transcription}\n\n")
    #         self.update_state(interrupt=True)
    #         if self.audio_stream_task and not self.audio_stream_task.done():
    #             self.audio_stream_task.cancel()
    #             self.openrouter_plugin.interrupt(self.agent_transcription)
    #         self.agent_transcription = ""

    #     # TODO: WE NEED to Stop chatgpt generation from conintuing
    #     elif self._agent_state == AgentState.THINKING:
    #         self.update_state(interrupt=True)
    #         if self.audio_stream_task and not self.audio_stream_task.done():
    #             self.audio_stream_task.cancel()
    #             self.openrouter_plugin.interrupt_with_user_message()

    # TODO better handeling of interuption
    def on_active_speakers_changed(self, speakers: list[rtc.Participant]):
        if speakers:
            active_speaker = speakers[0]
            logging.info(f"Active speaker: {active_speaker.identity}")
            self.audio_out_gain = 0.5
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
    
    def create_message_task(self, message: str, same_uterance: bool = False, speak_only: bool = False):
        # Stop all previous running process_user_chat_message jobs
        if self.user_chat_message_stop_events:
            logging.info("stoping thread")
            for stop_event in self.user_chat_message_stop_events:
                stop_event.set()

        # Create a new stop event for the current uterance
        stop_event = threading.Event()
        self.user_chat_message_stop_events.append(stop_event)

        # Start a new thread for processing the user chat message
        logging.info(f"Create new task: {message}")
        threading.Thread(target=self.process_user_chat_message, args=(message, same_uterance, stop_event, speak_only)).start()
    
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
            if self.localVideoTranscript == False:
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
    
    def process_user_audio_track(self, track):
        async def process_audio_stream():
            audio_stream = rtc.AudioStream(track)
            stream = self.user_stt_plugin.stream()
            logging.info("STARTED process_user_audio_track")
            self.ctx.create_task(self.process_user_stt_stream(stream))

            async for audio_frame_event in audio_stream:
                stream.push_frame(audio_frame_event.frame)

            await stream.flush()
            logging.info("STOPPED process_user_audio_track")

        def run_async_audio_stream():
            asyncio.run(process_audio_stream())

        threading.Thread(target=run_async_audio_stream).start()

    #TODO: is there a better way to handel timer?
    #TODO: better interuption logic
    async def process_user_stt_stream(self, stream):
        logging.info("STARTED process_user_stt_stream")
        buffered_text = ""
        same_uterance_timeout = 2 # Time in seconds in which to count stt result as the same uterance as previous result
        uterance_time = 0
        start_of_uterance = True
        is_first_uterance = True
        async for event in stream:
            if event.alternatives[0].text == "":
                continue
            elif start_of_uterance:
                uterance_time = time.time()
                start_of_uterance = False

            if event.is_final:
                buffered_text = " ".join([buffered_text, event.alternatives[0].text])

            if not event.end_of_speech:
                continue

            if buffered_text == "":
                continue
            
            # TODO this same uterance logic is broken
            same_uterance = False
            elapsed_time = time.time() - uterance_time
            if elapsed_time > same_uterance_timeout and uterance_time != 0 and not is_first_uterance:
                logging.info(f"Same uterance: {elapsed_time}")
                same_uterance = True
            else: logging.info(f"Diff uterance: {elapsed_time}")


            self.create_message_task(buffered_text, same_uterance)

            buffered_text = ""
            start_of_uterance = True
            is_first_uterance = False
        logging.info("STOPED process_user_stt_stream")

    # TODO we should have a finished event
    # TODO log if we are waiting on lock
    def process_user_chat_message(self, uterance: str, same_uterance: bool, stop_event: threading.Event, speak_only: bool = False):
        async def process_chat_message():
            tts = TTS(
            # api_key=os.getenv("DEEPGRAM_API_KEY", os.environ["DEEPGRAM_API_KEY"]),
            )
            try:
                if speak_only:
                    logging.info("Intro")
                    self.update_state(processing=True)
                    stream = tts.stream()
                    self.audio_out_gain = 1.0
                    msg = ChatGPTMessage(role=ChatGPTMessageRole.assistant, content=uterance)
                    self.openrouter_plugin._messages.append(msg)
                    send_audio_task = asyncio.create_task(self.send_audio_stream(stream, stop_event, False))
                    if not stop_event.is_set():
                        stream.push_text(uterance)
                        await stream.flush()
                    if not stop_event.is_set(): 
                        await send_audio_task

                # elif same_uterance and self.last_agent_message is not None:
                elif False:
                    logging.info("Updating Message")
                    self.update_state(processing=True)
                    stream = tts.stream()
                    self.audio_out_gain = 1.0

                    last_message_content = self.last_user_message.message
                    self.openrouter_plugin.interrupt_and_pop_user_message(last_message_content)
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
                        result = await self.process_chatgpt_result_return(chatgpt_stream, stop_event)
                    if not stop_event.is_set():
                        stream.push_text(result)
                        await stream.flush()
                    if not stop_event.is_set(): 
                        await send_audio_task

                else:
                    logging.info("New message")
                    self.update_state(processing=True)
                    stream = tts.stream()
                    self.audio_out_gain = 1.0

                    chat_message = ChatMessage(message=uterance)
                    # TODO: Write user chat update method
                    await self.ctx.room.local_participant.publish_data(
                        payload=json.dumps(chat_message.asjsondict()),
                        kind=DataPacketKind.KIND_RELIABLE,
                        topic="lk-chat-topic",
                    )

                    #TODO Fix interupt:
                    if self.last_agent_message:
                        self.openrouter_plugin.interrupt(self.last_agent_message.message)
                    # else:
                        # self.openrouter_plugin.interrupt()
                    msg = self.process_chatgpt_input(uterance)
                    self.last_user_message = chat_message
                    chatgpt_stream = self.openrouter_plugin.add_message(msg)
                    
                    send_audio_task = asyncio.create_task(self.send_audio_stream(stream, stop_event, False))
                    if not stop_event.is_set():
                        result = await self.process_chatgpt_result_return(chatgpt_stream, stop_event)
                    if not stop_event.is_set():
                        stream.push_text(result)                    
                        await stream.flush()
                    if not stop_event.is_set():
                        await send_audio_task

            except StopProcessingException:
                logging.info("process_user_chat_message stop event")
            except Exception as e:
                logging.error(f"An unexpected error occurred in process_user_chat_message: {e}", exc_info=True)
            finally:
                await stream.aclose()
                self.update_state(processing=False)
                logging.info("EXITED TASK: process chat message")

        def run_async_chat_message():
            with self.process_user_chat_lock:
                asyncio.set_event_loop(self.shared_event_loop)
                self.shared_event_loop.run_until_complete(process_chat_message())

        threading.Thread(target=run_async_chat_message).start()

    # TODO: create local log of all voice transcriptions 
    async def process_agent_stt_stream(self, stream):
        buffered_text = ""
        async for event in stream:
            if event.alternatives[0].text == "":
                continue

            if event.is_final:
                buffered_text = " ".join([buffered_text, event.alternatives[0].text])
                self.agent_transcription = " ".join([self.agent_transcription, event.alternatives[0].text])

            if not event.end_of_speech:
                continue
            
            if self.start_of_message:
                self.start_of_message = False
                self.last_agent_message = await self.chat.send_message(buffered_text)
            else:
                # Extract the "message" from self.last_agent_message
                last_message_content = self.last_agent_message.message

                # Update the message content with the new buffered_text
                updated_message_content = last_message_content + buffered_text

                # Update the "message" field of self.last_agent_message
                self.last_agent_message.message = updated_message_content

                # Send the updated message using self.chat.update_message
                await self.chat.update_message(self.last_agent_message)
            # await self.ctx.room.local_participant.publish_data(
            #     json.dumps(
            #         {
            #             "text": buffered_text,
            #             "timestamp": int(datetime.now().timestamp() * 1000),
            #             "speaker": "agent",
            #         }
            #     ),
            #     topic="transcription",
            # )
            buffered_text = ""
            
    def process_chatgpt_input(self, message):
        if self.video_enabled:
            last_entries = self.get_last_entries(5)
            user_message = "Summary of the last few frames: \n\n"  + last_entries + "\n\nUser Message: " + message
            if self.chatmodel_multimodal:
                msg = ChatGPTMessage(role=ChatGPTMessageRole.user, content=user_message, image_data=self.latest_frame, image_width=self.latest_frame_width, image_height=self.latest_frame_height)
            else:
                msg = ChatGPTMessage(role=ChatGPTMessageRole.user, content=user_message)
        else: 
            user_message = message
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

    async def process_chatgpt_result(self, text_stream, stop_event: threading.Event = None):
        logging.info("Process ChatGPT Result")
        self.audio_out_gain = 1.0
        # ChatGPT is streamed, so we'll flip the state immediately
        self.update_state(processing=True)

        stream = self.tts_plugin.stream()
        self.audio_stream_task = self.ctx.create_task(self.send_audio_stream(stream, stop_event))
        
        try:
            all_text = ""
            async for text in text_stream:
                if stop_event is not None and stop_event.is_set():
                    logging.info("STOP EVENT text stream")
                    break
                all_text += text
            
            if stop_event is not None and stop_event.is_set():
                logging.info("STOP EVENT text stream return")
                self.update_state(processing=False)
                return
            
            logging.info(all_text)
            
            self.agent_transcription = ""
            stream.push_text(all_text)

            # buffer up the entire response from ChatGPT before sending a chat message
            # await self.chat.send_message(all_text) #TODO uncomment this but we need to figure out how to both stream agent transcript and send the actually text from chat gpt
            await stream.flush()

        except Exception as e:
            logging.error(f"An error occurred while processing ChatGPT result: {e}", exc_info=True)
        finally:
            self.update_state(processing=False)

    async def process_chatgpt_result_return(self, text_stream, stop_event: threading.Event = None):
        try:
            all_text = ""
            async for text in text_stream:
                if stop_event is not None and stop_event.is_set():
                    logging.info("STOP EVENT process text stream")
                    raise StopProcessingException("Stop event set, halting text stream processing.")
                all_text += text
            
            if stop_event is not None and stop_event.is_set():
                logging.info("STOP EVENT process text stream")
                raise StopProcessingException("Stop event set, halting text stream processing.")
            
            logging.info(all_text)
            return(all_text)

        except Exception as e:
            logging.error(f"An error occurred while processing ChatGPT result: {e}", exc_info=True)
            raise e

    async def send_audio_stream(self, tts_stream: AsyncIterable[SynthesisEvent], stop_event: threading.Event = None, close_stream: bool = True):
        try:
            self.start_of_message = True
            async for e in tts_stream:
                if stop_event is not None and stop_event.is_set():
                    raise StopProcessingException("Stop event set, halting text stream processing.")

                if e.type == SynthesisEventType.STARTED:
                    self.update_state(sending_audio=True)
                elif e.type == SynthesisEventType.FINISHED:
                    self.update_state(sending_audio=False)
                    break
                elif e.type == SynthesisEventType.AUDIO:
                    if self._agent_state == AgentState.LISTENING:
                        # Stop the audio stream if the agent is listening
                        break
                
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
            state = AgentState.LISTENING
            self._agent_interupted = True
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
            self.start_of_message = True
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
        level=logging.INFO,
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
