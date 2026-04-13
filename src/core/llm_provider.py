"""
LLM Provider Chain for PolicyLens.

Implements auto-failover across three LLM providers:
    1. Groq API (Llama 3.3 70B) - fast, cloud, free tier
    2. Ollama (Mistral 7B) - local, no API key needed
    3. Flan-T5 (HuggingFace) - CPU, always available

The chain tries each provider in order. If one fails (network error,
rate limit, timeout), it automatically falls to the next. The system
never crashes due to LLM unavailability.

Usage:
    from src.core.llm_provider import llm_chain
    response = llm_chain.generate("Summarize this text: ...")
"""

import logging
import time
from typing import Optional

import httpx

from src.core.config import settings

logger = logging.getLogger("policylens.llm")


class LLMResponse:
    """Standardized response from any LLM provider.

    Wraps the raw provider response into a consistent format
    so downstream agents don't care which provider generated it.
    """

    def __init__(self, text: str, provider: str, model: str, latency: float, tokens_used: int = 0):
        self.text = text
        self.provider = provider
        self.model = model
        self.latency = latency
        self.tokens_used = tokens_used

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "provider": self.provider,
            "model": self.model,
            "latency_seconds": round(self.latency, 4),
            "tokens_used": self.tokens_used,
        }


class GroqProvider:
    """Groq API provider using Llama 3.3 70B.

    Fastest option (~200 tok/s). Requires GROQ_API_KEY in .env.
    Free tier has rate limits (~30 requests/minute).
    """

    def __init__(self):
        self.api_key = settings.groq_api_key
        self.model = "llama-3.3-70b-versatile"
        self.name = "groq"

        # Read model from config if available
        for provider_config in settings.llm_providers:
            if provider_config.get("name") == "groq":
                self.model = provider_config.get("model", self.model)
                break

    def is_available(self) -> bool:
        """Check if Groq is configured with an API key."""
        return bool(self.api_key and self.api_key.strip())

    def generate(self, prompt: str, max_tokens: int = 2048, temperature: float = 0.3) -> LLMResponse:
        """Generate text using Groq API.

        Args:
            prompt: The input prompt.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0=deterministic, 1=creative).

        Returns:
            LLMResponse with generated text and metadata.

        Raises:
            Exception: If API call fails.
        """
        from groq import Groq

        client = Groq(api_key=self.api_key)
        start = time.perf_counter()

        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )

        latency = time.perf_counter() - start
        text = response.choices[0].message.content
        tokens = response.usage.total_tokens if response.usage else 0

        return LLMResponse(
            text=text,
            provider=self.name,
            model=self.model,
            latency=latency,
            tokens_used=tokens,
        )


class OllamaProvider:
    """Ollama local provider using Mistral 7B.

    Runs locally, no API key needed. Requires Ollama to be
    installed and running (ollama serve). Moderate speed (~30 tok/s).
    """

    def __init__(self):
        self.base_url = settings.ollama_base_url
        self.model = "mistral:7b"
        self.name = "ollama"

        for provider_config in settings.llm_providers:
            if provider_config.get("name") == "ollama":
                self.model = provider_config.get("model", self.model)
                break

    def is_available(self) -> bool:
        """Check if Ollama server is running."""
        try:
            response = httpx.get(f"{self.base_url}/api/tags", timeout=2.0)
            return response.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

    def generate(self, prompt: str, max_tokens: int = 2048, temperature: float = 0.3) -> LLMResponse:
        """Generate text using Ollama local API.

        Args:
            prompt: The input prompt.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            LLMResponse with generated text.

        Raises:
            Exception: If Ollama server is unreachable or errors.
        """
        start = time.perf_counter()

        response = httpx.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature,
                },
                "stream": False,
            },
            timeout=120.0,
        )
        response.raise_for_status()

        latency = time.perf_counter() - start
        data = response.json()

        return LLMResponse(
            text=data.get("response", ""),
            provider=self.name,
            model=self.model,
            latency=latency,
            tokens_used=data.get("eval_count", 0),
        )


class FlanT5Provider:
    """HuggingFace Flan-T5 provider running on CPU.

    Always available — no internet, no server, no API key.
    Slowest and lowest quality, but guarantees the system
    never fails completely. Uses the base model (250MB).
    """

    def __init__(self):
        self.model_name = "google/flan-t5-base"
        self.name = "flan-t5"
        self._pipeline = None

        for provider_config in settings.llm_providers:
            if provider_config.get("name") == "flan-t5":
                self.model_name = provider_config.get("model", self.model_name)
                break

    def is_available(self) -> bool:
        """Flan-T5 is always available (runs on CPU)."""
        return True

    @property
    def pipeline(self):
        """Lazy-load the model pipeline on first use."""
        if self._pipeline is None:
            from transformers import pipeline as hf_pipeline
            logger.info("Loading Flan-T5 model: %s (this may take a moment)", self.model_name)
            self._pipeline = hf_pipeline(
                "text2text-generation",
                model=self.model_name,
                device=-1,  # Force CPU
            )
            logger.info("Flan-T5 model loaded successfully")
        return self._pipeline

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.3) -> LLMResponse:
        """Generate text using Flan-T5 on CPU.

        Args:
            prompt: The input prompt.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            LLMResponse with generated text.
        """
        start = time.perf_counter()

        # Flan-T5 base has a 512 token input limit, truncate if needed
        truncated_prompt = prompt[:2048]

        outputs = self.pipeline(
            truncated_prompt,
            max_new_tokens=min(max_tokens, 512),
            temperature=max(temperature, 0.01),
            do_sample=temperature > 0,
        )

        latency = time.perf_counter() - start
        text = outputs[0]["generated_text"]

        return LLMResponse(
            text=text,
            provider=self.name,
            model=self.model_name,
            latency=latency,
            tokens_used=len(text.split()),
        )


class LLMProviderChain:
    """Auto-failover chain across multiple LLM providers.

    Tries each provider in priority order. If one fails, logs the
    error and tries the next. Only raises an exception if ALL
    providers fail.
    """

    def __init__(self):
        self.providers = [
            GroqProvider(),
            OllamaProvider(),
            FlanT5Provider(),
        ]
        self._last_provider = None

    def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.3,
        preferred_provider: Optional[str] = None,
    ) -> LLMResponse:
        """Generate text using the first available provider.

        Args:
            prompt: The input prompt.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            preferred_provider: Optionally force a specific provider by name.

        Returns:
            LLMResponse from whichever provider succeeded.

        Raises:
            RuntimeError: If all providers fail.
        """
        providers_to_try = self.providers

        # If a specific provider is requested, try it first
        if preferred_provider:
            preferred = [p for p in self.providers if p.name == preferred_provider]
            others = [p for p in self.providers if p.name != preferred_provider]
            providers_to_try = preferred + others

        errors = []
        for provider in providers_to_try:
            # Check availability before attempting
            if not provider.is_available():
                logger.info(
                    "Provider '%s' is not available, skipping",
                    provider.name,
                )
                continue

            try:
                logger.info("Trying provider: %s (%s)", provider.name, provider.model)
                response = provider.generate(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                self._last_provider = provider.name
                logger.info(
                    "Provider '%s' succeeded in %.2fs (%d tokens)",
                    provider.name,
                    response.latency,
                    response.tokens_used,
                )
                return response

            except Exception as e:
                error_msg = f"{provider.name}: {str(e)}"
                errors.append(error_msg)
                logger.warning(
                    "Provider '%s' failed: %s. Trying next provider.",
                    provider.name,
                    str(e),
                )

        # All providers failed
        error_summary = "; ".join(errors)
        raise RuntimeError(
            f"All LLM providers failed. Errors: {error_summary}"
        )

    def get_status(self) -> list[dict]:
        """Check availability of all providers.

        Returns:
            List of provider status dicts for dashboard display.
        """
        status = []
        for provider in self.providers:
            status.append({
                "name": provider.name,
                "model": provider.model if hasattr(provider, "model") else provider.model_name,
                "available": provider.is_available(),
            })
        return status

    @property
    def last_provider(self) -> Optional[str]:
        """Name of the provider used in the most recent generation."""
        return self._last_provider


# Singleton instance
llm_chain = LLMProviderChain()