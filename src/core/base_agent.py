"""
Base agent interface for PolicyLens multi-agent system.

All agents inherit from BaseAgent and implement the process() method.
This enforces a consistent contract across all 15 agents, enabling
the orchestrator to call any agent through the same interface.

Design Pattern: Template Method
    execute() defines the skeleton (validate -> process -> return)
    Subclasses override process() with their specific logic.
"""

import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone


class BaseAgent(ABC):
    """Abstract base class that every PolicyLens agent must inherit from.

    Provides automatic logging, execution timing, and health checks.
    Subclasses only need to implement process() with their domain logic.
    """
#Constructor
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.logger = logging.getLogger(f"policylens.agent.{name}") 
        self._last_execution_time = None
        self._execution_count = 0
#Contract
    @abstractmethod
    def process(self, input_data: dict) -> dict:
        """Core logic that each agent implements.

        Args:
            input_data: Dictionary containing agent-specific input.

        Returns:
            Dictionary containing agent-specific output.
        """
        pass
#Template Method
    def execute(self, input_data: dict) -> dict:
        """Template method that wraps process() with logging and timing.

        This method is NOT meant to be overridden. It provides the
        consistent execution skeleton that the orchestrator relies on.

        Args:
            input_data: Dictionary containing agent-specific input.

        Returns:
            Dictionary containing agent output plus execution metadata.
        """
        self.logger.info("Agent '%s' starting execution", self.name)
        start_time = time.perf_counter()

        try:
            result = self.process(input_data)
            elapsed = time.perf_counter() - start_time
            self._last_execution_time = elapsed
            self._execution_count += 1

            self.logger.info(
                "Agent '%s' completed in %.3f seconds",
                self.name,
                elapsed,
            )

            result["_metadata"] = {
                "agent": self.name,
                "execution_time_seconds": round(elapsed, 4),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            return result

        except Exception as e:
            elapsed = time.perf_counter() - start_time
            self.logger.error(
                "Agent '%s' failed after %.3f seconds: %s",
                self.name,
                elapsed,
                str(e),
            )
            raise
#health check
    def health_check(self) -> dict:
        """Report agent health status for the monitoring dashboard.

        Returns:
            Dictionary with agent name, status, and execution stats.
        """
        return {
            "agent": self.name,
            "status": "healthy",
            "description": self.description,
            "executions": self._execution_count,
            "last_execution_time": self._last_execution_time,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"