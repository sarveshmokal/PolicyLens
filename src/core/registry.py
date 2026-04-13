"""
Agent Registry for PolicyLens.

Dynamically discovers, loads, and manages agents based on the
agents.yaml configuration. Agents are loaded on-demand (lazy loading)
and cached after first instantiation.

This enables the plugin architecture: add a new agent by creating
a class that inherits BaseAgent and adding an entry to agents.yaml.
No other code changes required.
"""

import importlib
import logging
from typing import Optional

from src.core.base_agent import BaseAgent
from src.core.config import settings

logger = logging.getLogger("policylens.registry")


class AgentRegistry:
    """Central registry that manages all PolicyLens agents.

    Reads agent definitions from agents.yaml, dynamically imports
    agent classes, and provides lookup by name. Agents are instantiated
    lazily on first access and cached for subsequent calls.
    """

    def __init__(self):
        self._agents: dict[str, BaseAgent] = {}
        self._agent_configs: dict[str, dict] = {}
        self._load_agent_configs()

    def _load_agent_configs(self) -> None:
        """Parse agents.yaml and build a flat lookup of agent configs.

        Converts the nested group/agent structure into a flat dict
        keyed by agent name for fast lookup.
        """
        agents_config = settings.agents
        for group_name, group_agents in agents_config.items():
            for agent_name, agent_config in group_agents.items():
                self._agent_configs[agent_name] = {
                    "group": group_name,
                    **agent_config,
                }
        logger.info(
            "Loaded %d agent configurations from agents.yaml",
            len(self._agent_configs),
        )

    def _import_agent_class(self, module_path: str, class_name: str) -> type:
        """Dynamically import an agent class from its module path.

        Args:
            module_path: Dotted module path (e.g., 'src.agents.analysis.retriever')
            class_name: Class name within the module (e.g., 'RetrieverAgent')

        Returns:
            The agent class (not an instance).

        Raises:
            ImportError: If the module cannot be found.
            AttributeError: If the class doesn't exist in the module.
        """
        module = importlib.import_module(module_path)
        agent_class = getattr(module, class_name)

        if not issubclass(agent_class, BaseAgent):
            raise TypeError(
                f"{class_name} in {module_path} does not inherit from BaseAgent"
            )

        return agent_class

    def get(self, agent_name: str) -> BaseAgent:
        """Get an agent instance by name, loading it if necessary.

        First call imports and instantiates the agent. Subsequent
        calls return the cached instance.

        Args:
            agent_name: Agent identifier as defined in agents.yaml.

        Returns:
            Instantiated agent object.

        Raises:
            KeyError: If agent name not found in config.
            RuntimeError: If agent is disabled in config.
            ImportError: If agent module cannot be loaded.
        """
        # Return cached instance if already loaded
        if agent_name in self._agents:
            return self._agents[agent_name]

        # Check if agent exists in config
        if agent_name not in self._agent_configs:
            raise KeyError(
                f"Unknown agent: '{agent_name}'. "
                f"Available: {list(self._agent_configs.keys())}"
            )

        config = self._agent_configs[agent_name]

        # Check if agent is enabled
        if not config.get("enabled", False):
            raise RuntimeError(
                f"Agent '{agent_name}' is disabled in agents.yaml. "
                f"Set enabled: true to use it."
            )

        # Dynamic import and instantiation
        agent_class = self._import_agent_class(
            config["module"], config["class"]
        )
        agent_instance = agent_class(
            name=agent_name,
            description=config.get("description", ""),
        )

        # Cache for future calls
        self._agents[agent_name] = agent_instance
        logger.info("Loaded agent: %s (%s)", agent_name, config["class"])

        return agent_instance

    def get_group(self, group_name: str) -> list[BaseAgent]:
        """Load all enabled agents in a group, in config order.

        Args:
            group_name: Group identifier ('ingestion', 'analysis', etc.)

        Returns:
            List of instantiated agents in the order defined in agents.yaml.
        """
        agents = []
        for agent_name, config in self._agent_configs.items():
            if config["group"] == group_name and config.get("enabled", True):
                try:
                    agents.append(self.get(agent_name))
                except (ImportError, AttributeError) as e:
                    logger.warning(
                        "Skipping agent '%s': %s", agent_name, str(e)
                    )
        return agents

    def list_available(self) -> list[dict]:
        """List all registered agents with their status.

        Returns:
            List of dicts with agent name, group, enabled status, and
            whether the agent is currently loaded in memory.
        """
        result = []
        for name, config in self._agent_configs.items():
            result.append({
                "name": name,
                "group": config["group"],
                "enabled": config.get("enabled", False),
                "loaded": name in self._agents,
                "description": config.get("description", ""),
            })
        return result

    def health_check_all(self) -> list[dict]:
        """Run health checks on all loaded agents.

        Returns:
            List of health check responses from each loaded agent.
        """
        results = []
        for name, agent in self._agents.items():
            try:
                results.append(agent.health_check())
            except Exception as e:
                results.append({
                    "agent": name,
                    "status": "unhealthy",
                    "error": str(e),
                })
        return results


# Singleton instance
registry = AgentRegistry()
