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
import logging
import asyncio
import openai
from dataclasses import dataclass
from typing import AsyncIterable, List, Optional
from enum import Enum
import base64
import io
from PIL import Image

ChatGPTMessageRole = Enum("MessageRole", ["system", "user", "assistant", "function"])

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

@dataclass
class ChatGPTMessage:
    role: ChatGPTMessageRole
    content: str
    image_data: Optional[bytes] = None
    image_width: Optional[int] = None
    image_height: Optional[int] = None

    def to_api(self, process_image=False):
        message = {"role": self.role.name}

        if self.image_data is not None and process_image:
            image_rgba = Image.frombytes(
                "RGBA", (self.image_width, self.image_height), self.image_data
            )
            image_rgb = image_rgba.convert("RGB")
            jpg_img = io.BytesIO()
            image_rgb.save(jpg_img, format="JPEG")
            base64_img = base64.b64encode(jpg_img.getvalue()).decode('utf-8')
            message["content"] = [
                {"type": "text", "text": self.content},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
            ]
        else:
            message["content"] = self.content

        return message


class ChatGPTPlugin:
    """OpenAI ChatGPT Plugin"""

    def __init__(self, prompt: str, message_capacity: int, model: str, api_key: str, base_url: Optional[str] = None):
        """
        Args:
            prompt (str): First 'system' message sent to the chat that prompts the assistant
            message_capacity (int): Maximum number of messages to send to the chat
            model (str): Which model to use (i.e. 'gpt-3.5-turbo')
        """
        self._model = model
        self._client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._prompt = prompt
        self._message_capacity = message_capacity
        self._messages: List[ChatGPTMessage] = []
        self._producing_response = False
        self._needs_interrupt = False
        self._save_response = True

    def clear_history(self):
        """Clears all messages from the chat history."""
        self._messages = []   

    def interrupt(self):
        """Interrupt a currently streaming response (if there is one)"""
        if self._producing_response:
            self._needs_interrupt = True
    
    def interrupt(self, new_text: str):
        """Interrupt a currently streaming response (if there is one) and replace the last assistant message with the provided text"""
        if self._producing_response:
            self._needs_interrupt = True
            print("interupting chatgpt stream")

        # Replace the assitant message with the text we actually spoke.
        if self._messages and self._messages[-1].role == ChatGPTMessageRole.assistant:
            print("Replaceing agent message")
            self._messages[-1].content = new_text

    def interrupt_with_user_message(self, append_text: Optional[str] = None):
        """Interrupt a currently streaming response (if there is one) and clear assistant message"""
        if self._producing_response:
            self._save_response = False
            # self._needs_interrupt = True
        
        else:
            if self._messages and self._messages[-1].role == ChatGPTMessageRole.assistant:
                self._messages.pop()
    
    def interrupt_and_pop_user_message(self, text: str):
        """Interrupt an ongoing response generation and remove the user message containing the specified text, along with all subsequent messages."""
        if self._producing_response:
            self._needs_interrupt = True
            self._save_response = False
            print("interupting chatgpt stream")

        # Check the last 2 user messages
        for i in range(len(self._messages)-1, -1, -1):
            message = self._messages[i]
            if isinstance(message, ChatGPTMessage) and message.role == ChatGPTMessageRole.user:
                if isinstance(message.content, str) and text in message.content:
                    # If the append_text is found in the user message content, 
                    # remove this message and all messages after it
                    self._messages = self._messages[:i]
                    break
                
                if isinstance(message.content, list):
                    for item in message.content:
                        if item["type"] == "text" and text in item["text"]:
                            # If the append_text is found in the user message content,
                            # remove this message and all messages after it 
                            self._messages = self._messages[:i]
                            break
                    
                    # Break out of the outer loop if append_text was found
                    if len(self._messages) == i:
                        break
            elif isinstance(message, str) and text in message:
                # If the message is a string and append_text is found in it,
                # remove this message and all messages after it
                self._messages = self._messages[:i]
                break

    async def send_system_prompt(self) -> AsyncIterable[str]:
        """Send the system prompt to the chat and generate a streamed response

        Returns:
            AsyncIterable[str]: Streamed ChatGPT response
        """
        try:
            async for text in self.add_message(None):
                yield text
        except TimeoutError:
            yield "Sorry, I'm taking too long to respond. Please try again later."
            return

    async def add_message(
        self, message: Optional[ChatGPTMessage], save_response: bool = True
    ) -> AsyncIterable[str]:
        """Add a message to the chat and generate a streamed response

        Args:
            message (ChatGPTMessage): The message to add

        Returns:
            AsyncIterable[str]: Streamed ChatGPT response
        """

        if message is not None:
            self._messages.append(message)
        if len(self._messages) > self._message_capacity:
            self._messages.pop(0)

        async for text in self._generate_text_streamed(self._model, save_response=save_response):
            yield text

    async def query(
            self, message: ChatGPTMessage, system_prompt: Optional[str]
        ) -> AsyncIterable[str]:
            """Add a message to the chat and generate a streamed response

            Args:
                message (ChatGPTMessage): The message to add

            Returns:
                AsyncIterable[str]: Streamed ChatGPT response
            """

            if len(self._messages) + 1 > self._message_capacity:
                self._messages.pop(0)

            async for text in self._generate_text_streamed(self._model, message=message, system_message=system_prompt, save_response=False):
                yield text

    async def _generate_text_streamed(self, model: str, message: ChatGPTMessage = None, system_message: str = None, save_response: bool = True) -> AsyncIterable[str]:
        if system_message is not None:
            prompt_message = ChatGPTMessage(
                role=ChatGPTMessageRole.system, content=system_message
            )
        else:
            prompt_message = ChatGPTMessage(
                role=ChatGPTMessageRole.system, content=self._prompt
            )
        try:
            chat_messages = [
                m.to_api(process_image=(i == len(self._messages) - 1))
                for i, m in enumerate(self._messages)
            ]

            if message is not None:
                chat_messages.append(message.to_api())
            
            chat_stream = await asyncio.wait_for(
                self._client.chat.completions.create(
                    model=model,
                    n=1,
                    stream=True,
                    messages=[prompt_message.to_api()] + chat_messages,
                    max_tokens=320000,
                ),
                600,
            )
        except TimeoutError:
            yield "Sorry, I'm taking too long to respond. Please try again later."
            return

        self._producing_response = True
        complete_response = ""

        async def anext_util(aiter):
            async for item in aiter:
                return item

            return None

        while True:
            try:
                chunk = await asyncio.wait_for(anext_util(chat_stream), 5)
            except TimeoutError:
                break
            except asyncio.CancelledError:
                self._producing_response = False
                self._needs_interrupt = False
                break

            if chunk is None:
                break
            content = chunk.choices[0].delta.content

            if self._needs_interrupt:
                self._needs_interrupt = False
                logging.info("ChatGPT interrupted")
                break

            if content is not None:
                complete_response += content
                yield content

        if self._save_response and save_response:
            self._messages.append(
                ChatGPTMessage(role=ChatGPTMessageRole.assistant, content=complete_response)
            )

        self._save_response = True
        self._producing_response = False

    def set_model(self, model: str):
        """Change the model used by the ChatGPT plugin

        Args:
            model (str): The new model to use (e.g., 'gpt-3.5-turbo', 'gpt-4')
        """
        self._model = model

    def prompt(self):
        return self._prompt

    def prompt(self, new_prompt: str):
        self._prompt = new_prompt

    def get_chat_history(self, last_x_chats: Optional[int] = None) -> str:
        chat_history = ""
        messages_to_process = self._messages[-last_x_chats:] if last_x_chats is not None else self._messages
        for message in messages_to_process:
            role = "user" if message.role == ChatGPTMessageRole.user else "assistant"
            content = message.content
            
            if isinstance(content, list):
                content = "\n".join(item["text"] for item in content if item["type"] == "text")
            
            chat_history += f"[{role}]: {content}\n\n"
        
        return chat_history.strip()