from abc import ABC, abstractmethod

class BaseAgent(ABC):
    """Base class for all agents"""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    async def _execute(self, *args, **kwargs):
        raise NotImplementedError("Agents must implement the _execute method")
