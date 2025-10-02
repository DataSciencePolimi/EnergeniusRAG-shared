"""Guru class for orchestrating the workflow of the application."""

from langchain_core.messages import BaseMessage

from knowledge_base import KnowledgeManager
from llm import LLMHandler


class Guru:
    """Guru class for orchestrating the workflow of the application.

    This class is responsible for managing the flow of the application,
    including loading configurations, initializing components, and
    orchestrating the execution of tasks.
    """
    user_type: str
    house_type: str
    region: str

    def __init__(self, provider: str, model: str, embedding: str, language: str,
                 temperature: float, user_type: str = None, house_type: str = None, region: str = None, use_knowledge: bool = True) -> None:
        """Initialize the Guru class.

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
        self.llm = LLMHandler(
            provider=provider,
            model=model,
            temperature=temperature,
            language=language,
            keep_history=True
        )
        self.know_base = KnowledgeManager(
            provider=provider,
            model=model,
            embedding=embedding,
            language=language
        )
        self.user_knowledge = use_knowledge
        self.user_type = user_type
        self.house_type = house_type
        self.region = region

    def load_past_messages(self, messages: list[BaseMessage]) -> None:
        """
        Load past messages into the orchestrator.
        Args:
            messages (list[BaseMessage]): List of past messages to load.
        """
        self.llm.load_messages(messages)

    def user_message(self, message: str):# -> str:
        """
        Process a user message and return a response.
        Args:
            message (str): The user message to process.
        Returns:
            str: The response from the LLM.
        """
        if self.user_knowledge:
            #return self.llm.generate_response(self.know_base.user_message(message, self.user_type, self.house_type, self.region), message, False)
            yield from self.llm.generate_response_stream(self.know_base.user_message(message, self.user_type, self.house_type, self.region), message, False)
        else:
            #return self.llm.generate_response(None, message)
            yield from self.llm.generate_response_stream(None, message)
        
    def set_language(self, language: str) -> None:
        self.llm.set_language(language)
        
