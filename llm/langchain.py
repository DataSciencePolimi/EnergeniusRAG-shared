"""LangChain LLM wrapper."""

import os

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage

from private_settings import PRIVATE_SETINGS


class LLMHandler:
    """
    LLMHandler class for managing the LLM model.
    """

    model: BaseChatModel
    language: str
    messages: list = []

    def __init__(
        self,
        provider: str,
        model: str,
        temperature: float,
        language: str,
        keep_history: bool = True,
    ) -> None:
        """Initialize the LLMHandler with the model name, provider, and API key.

        Args:
            provider (str): _model provider (e.g., "openai", "llama", etc.)._
            model (str): _model name (e.g., "gpt-3.5-turbo", "llama-2", "claude-2")._
            temperature (float): _temperature for the model (0.0 to 1.0)._
            language (str): _language for the model (e.g., "English", "Italiano", etc.)._
            keep_history (bool): _whether to keep the message history or not._
        """
        # storing the parameters
        self.language = language
        self.keep_history = keep_history

        # creation of the environment
        self.__env_creation(provider)

        # creation of the model
        if not PRIVATE_SETINGS["LLM_LOCAL"]:
            self.model = init_chat_model(
                model=model,
                model_provider=provider,
                temperature=temperature,
            )
        else:
            self.model = init_chat_model(
                model=model,
                model_provider=provider,
                temperature=temperature,
                base_url=PRIVATE_SETINGS["LLM_BASE_URL"],
            )

    def __env_creation(self, provider: str) -> None:
        """
        Create the environment variables for the LLM model.

        Args:
            provider (str): _Description of the model provider._
            api_key (str): _Description of the model provider API key._
        """

        if provider == "openai":
            os.environ["OPENAI_API_KEY"] = PRIVATE_SETINGS["LLM_KEY"]["openai"]

    def get_model(self) -> BaseChatModel:
        """Get the current BaseChatModel model
        Returns:
            BaseChatModel: _The current LLM model in use_
        """
        return self.model

    def set_model(self, model: BaseChatModel) -> None:
        """Set the new BaseChatModel model

        Args:
            model (BaseChatModel): _The new model to use_
        """
        self.model = model

    def get_language(self) -> str:
        """Get the current language of the LLMHandler

        Returns:
            str: _Current language_
        """
        return self.language

    def set_language(self, l: str) -> None:
        """Set new current language of the LLMHandler

        Args:
            l (str): _String of the language to set_
        """
        self.language = l

    def get_default_system_prompt(self) -> SystemMessage:
        """
        Get the default system prompt for the LLM model.

        Returns:
            SystemMessage: The default system message for the LLM model.
        """
        return SystemMessage(
            content=f"You are a helpful assistant that speaks {self.language}."
        )

    def load_messages(self, messages: list[BaseMessage]) -> None:
        """
        Load messages into the LLM model.

        Args:
            messages (list[BaseMessage]): List of past messages to load.
        """
        # Put in front ot the list the system message
        system_message = self.get_default_system_prompt()
        messages.insert(0, system_message)
        # Load the messages into the model
        self.messages = messages

    def clear_messages(self) -> None:
        """
        Clear the loaded messages.
        """
        self.messages = []

    def generate_response(
        self, system_prompt: str, user_message: str, use_past_history: bool = True
    ) -> str:
        """
        Generate a response from the LLM model based on the provided prompt.

        Args:
            prompt (str): The input prompt for the LLM model.

        Returns:
            str: The generated response from the LLM model.
        """
        # Check if the system prompt is None
        if system_prompt is None:
            system_prompt = self.get_default_system_prompt().content

        # Creation of the messages
        if use_past_history:
            # if the user wants to use the past history
            messages = self.messages.copy()
        else:
            messages = [SystemMessage(content=system_prompt)]

        user_message = HumanMessage(content=user_message)
        messages.append(user_message)

        # Generation of the response
        response: AIMessage = self.model.invoke(messages)

        # Check if messages should be kept
        if self.keep_history:
            self.messages.append(user_message)
            self.messages.append(response)

        return response.content

    def generate_response_stream(
        self, system_prompt: str, user_message: str, use_past_history: bool = True
    ):# -> str:
        """
        Generate a response from the LLM model based on the provided prompt.

        Args:
            prompt (str): The input prompt for the LLM model.

        Returns:
            str: The generated response from the LLM model.
        """
        # Check if the system prompt is None
        if system_prompt is None:
            system_prompt = self.get_default_system_prompt().content

        # Creation of the messages
        if use_past_history:
            # if the user wants to use the past history
            messages = self.messages.copy()
        else:
            messages = [SystemMessage(content=system_prompt)]

        user_message = HumanMessage(content=user_message)
        messages.append(user_message)

        # Generation of the response
        #response: AIMessage = self.model.invoke(messages)

        # Streaming: assume model has been initialized with streaming=True
        full_response = ""
        streamed_response = self.model.stream(messages)
        for chunk in streamed_response:
            if hasattr(chunk, "content"):
                token = chunk.content
            else:
                token = str(chunk)  # fallback, just in case

            full_response += token
            yield token  # stream each token

        # Check if messages should be kept
        if self.keep_history:
            self.messages.append(user_message)
            self.messages.append(full_response)

        #return response.content
