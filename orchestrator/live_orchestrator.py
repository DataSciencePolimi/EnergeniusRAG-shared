"""Orchestrator module for handling live user messages."""
from langchain_core.messages import AIMessage, HumanMessage

from .abstract_orchestrator import AbstractOrchestrator
from .guru import Guru


class LiveOrchestrator(AbstractOrchestrator):
    """
    Orchestrator for live chat.
    """

    def __init__(self, provider: str, model: str, embedding: str, temperature: float,
                 language: str, user_type: str, house_type: str, region: str, use_knowledge: bool = True) -> None:
        """Initialize the LiveOrchestrator with the model name, provider, and API key.

        Args:
            provider (str): _model provider (e.g., "openai", "llama", etc.)._
            model (str): _model name (e.g., "gpt-3.5-turbo", "llama-2", "claude-2")._
            embedding (str): _embedding model name (e.g., "text-embedding-3-small", etc.)._
            temperature (float): _temperature for the model (0.0 to 1.0)._
            language (str): _language for the model (e.g., "English", "Italiano", etc.)._
            user_type (str): _type of user (e.g., "working class", etc.)._
            house_type (str): _type of house (e.g., "apartment", "villa", etc.)._
            use_knowledge (bool): _whether to use the knowledge base or not._
        """
        self.guru = Guru(
            provider=provider,
            model=model,
            embedding=embedding,
            language=language,
            temperature=temperature,
            user_type=user_type,
            house_type=house_type,
            region=region,
            use_knowledge=use_knowledge
        )
       
    def load_past_messages(self, messages: list) -> None:
        """
        Load past messages into the orchestrator.
        """
        typed_list = []
        for m in messages:
            if m["role"] == "user":
                typed_list.append(HumanMessage(content=m["content"]))
            elif m["role"] == "assistant":
                typed_list.append(AIMessage(content=m["content"]))

        self.guru.load_past_messages(typed_list)

    def user_message(self, message: str):# -> str:
        """
        Process a user message and return a response.
        """
        #return self.guru.user_message(message)
        yield from self.guru.user_message_stream(message)
