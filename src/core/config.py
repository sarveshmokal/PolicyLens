"""
Configuration loader for PolicyLens.

Loads settings from three sources in priority order:
    1. Environment variables (.env) - secrets, overrides
    2. settings.yaml - application behavior
    3. agents.yaml - agent registry

Environment variables override YAML values when both exist.
This follows the 12-Factor App methodology.
"""

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv


# Project root is two levels up from this file (src/core/config.py -> PolicyLens/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Load .env file into os.environ
load_dotenv(PROJECT_ROOT / ".env")


def load_yaml(filename: str) -> dict:
    """Load a YAML config file from the config/ directory.

    Args:
        filename: Name of the YAML file (e.g., 'settings.yaml').

    Returns:
        Dictionary of parsed YAML content.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
    """
    filepath = PROJECT_ROOT / "config" / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Config file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class Settings:
    """Central configuration object for PolicyLens.

    Loads all config sources on initialization and provides
    typed access to settings throughout the application.
    """

    def __init__(self):
        # Load YAML configs
        self._settings = load_yaml("settings.yaml")
        self._agents = load_yaml("agents.yaml")

        # Environment overrides (env vars take priority over YAML)
        self.debug = os.getenv("DEBUG", "false").lower() == "true"
        self.log_level = os.getenv("LOG_LEVEL", "INFO").upper()

        # LLM Keys
        self.groq_api_key = os.getenv("GROQ_API_KEY", "")
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        # Redis
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    @property
    def chunking(self) -> dict:
        """Document chunking settings."""
        return self._settings.get("chunking", {})

    @property
    def embedding(self) -> dict:
        """Embedding model settings."""
        return self._settings.get("embedding", {})

    @property
    def retrieval(self) -> dict:
        """Retrieval pipeline settings."""
        return self._settings.get("retrieval", {})

    @property
    def verification(self) -> dict:
        """NLI verification settings."""
        return self._settings.get("verification", {})

    @property
    def llm_providers(self) -> list:
        """LLM provider chain configuration."""
        return self._settings.get("llm", {}).get("providers", [])

    @property
    def logging_config(self) -> dict:
        """Logging settings."""
        return self._settings.get("logging", {})

    @property
    def agents(self) -> dict:
        """Agent registry configuration."""
        return self._agents

    def get_agent_config(self, group: str, name: str) -> dict:
        """Get configuration for a specific agent.

        Args:
            group: Agent group ('ingestion', 'analysis', etc.)
            name: Agent name within the group ('retriever', 'planner', etc.)

        Returns:
            Agent configuration dict with enabled, module, class, description.

        Raises:
            KeyError: If agent group or name not found.
        """
        if group not in self._agents:
            raise KeyError(f"Unknown agent group: {group}")
        if name not in self._agents[group]:
            raise KeyError(f"Unknown agent '{name}' in group '{group}'")
        return self._agents[group][name]

    def is_agent_enabled(self, group: str, name: str) -> bool:
        """Check if a specific agent is enabled in config."""
        try:
            config = self.get_agent_config(group, name)
            return config.get("enabled", False)
        except KeyError:
            return False


# Singleton instance - import this throughout the application
# Usage: from src.core.config import settings
settings = Settings()