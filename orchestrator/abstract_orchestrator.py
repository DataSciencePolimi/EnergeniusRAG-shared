"""Orchestrator Abstract module definition for handling user messages."""
from abc import ABC, abstractmethod


class AbstractOrchestrator(ABC):
    """
    Abstract base class for orchestrators.
    """

    @abstractmethod
    def user_message(self, message: str) -> str:
        """
        Process a user message and return a response.
        """
