"""
Guardrails Agent for PolicyLens.

Safety and content filtering layer. Checks for PII (personal
identifiable information), prompt injection attempts, and
content quality issues in both inputs and outputs.

Runs as a pre-check on queries and post-check on answers
to ensure the system doesn't leak sensitive information
or respond to malicious prompts.
"""

import re

from src.core.base_agent import BaseAgent


class GuardrailsAgent(BaseAgent):
    """Content safety and quality filtering.

    Detects PII patterns (emails, phone numbers, SSNs, credit cards),
    prompt injection attempts, and content quality issues.
    Can operate on both input queries and output answers.
    """

    def __init__(self, name: str = "guardrails", description: str = ""):
        super().__init__(name=name, description=description)

        # PII detection patterns
        self.pii_patterns = {
            "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
            "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
            "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
            "credit_card": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
            "ip_address": r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
        }

        # Prompt injection patterns
        self.injection_patterns = [
            r"ignore\s+(previous|above|all)\s+instructions",
            r"disregard\s+(previous|above|all)",
            r"you\s+are\s+now\s+",
            r"act\s+as\s+(if|though)",
            r"pretend\s+(you|to\s+be)",
            r"forget\s+(everything|all|previous)",
            r"system\s*prompt",
            r"jailbreak",
        ]

    def process(self, input_data: dict) -> dict:
        """Run safety checks on input text and/or output text.

        Args:
            input_data: Must contain at least one of:
                - 'query' (str): user input to check
                - 'answer' (str): generated output to check

        Returns:
            Dictionary with safety verdict, detected issues, and
            optionally redacted text.
        """
        query = input_data.get("query", "")
        answer = input_data.get("answer", "")

        issues = []

        # Check query for injection attempts
        if query:
            injection_found = self._detect_injection(query)
            if injection_found:
                issues.extend(injection_found)

        # Check both query and answer for PII
        for label, text in [("query", query), ("answer", answer)]:
            if text:
                pii_found = self._detect_pii(text)
                for pii in pii_found:
                    pii["source"] = label
                    issues.append(pii)

        # Redact PII from answer if found
        redacted_answer = self._redact_pii(answer) if answer else ""

        # Determine safety verdict
        has_injection = any(i["type"] == "injection" for i in issues)
        has_pii = any(i["type"] == "pii" for i in issues)

        if has_injection:
            verdict = "BLOCKED"
        elif has_pii:
            verdict = "REDACTED"
        else:
            verdict = "CLEAN"

        self.logger.info(
            "Guardrails check: verdict=%s, issues=%d",
            verdict,
            len(issues),
        )

        return {
            "verdict": verdict,
            "issues": issues,
            "issues_count": len(issues),
            "has_injection": has_injection,
            "has_pii": has_pii,
            "redacted_answer": redacted_answer,
            "original_answer": answer,
        }

    def _detect_pii(self, text: str) -> list[dict]:
        """Detect PII patterns in text.

        Args:
            text: Text to scan for PII.

        Returns:
            List of detected PII items with type and matched text.
        """
        found = []
        for pii_type, pattern in self.pii_patterns.items():
            matches = re.findall(pattern, text)
            for match in matches:
                found.append({
                    "type": "pii",
                    "pii_type": pii_type,
                    "matched": match,
                })
        return found

    def _detect_injection(self, text: str) -> list[dict]:
        """Detect prompt injection attempts in input text.

        Args:
            text: User input to scan.

        Returns:
            List of detected injection patterns.
        """
        found = []
        text_lower = text.lower()
        for pattern in self.injection_patterns:
            if re.search(pattern, text_lower):
                found.append({
                    "type": "injection",
                    "pattern": pattern,
                    "severity": "high",
                })
        return found

    def _redact_pii(self, text: str) -> str:
        """Replace detected PII with redaction markers.

        Args:
            text: Text containing PII to redact.

        Returns:
            Text with PII replaced by [REDACTED_TYPE] markers.
        """
        redacted = text
        for pii_type, pattern in self.pii_patterns.items():
            redacted = re.sub(pattern, f"[REDACTED_{pii_type.upper()}]", redacted)
        return redacted