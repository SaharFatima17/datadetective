"""LLM access layer (proposal Sec.8).

The LLM is used ONLY for planning, hypothesis phrasing, critique and narrative.
It never produces a number that ends up in a finding - those come from the tool
layer. See Sec.3 ("trusted calculation") and Sec.13 ("grounding, not invention").

The `mock` provider lets the whole system run end-to-end with no API key, which
is what you use while building and testing everything below Phase 7.
"""

from __future__ import annotations

import json
import re

from app.config import settings

DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-4-5",
    "openai": "gpt-4o-mini",
    "gemini": "gemini-2.0-flash",
}


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(self) -> None:
        self.provider = settings.LLM_PROVIDER.lower()
        self.model = settings.LLM_MODEL or DEFAULT_MODELS.get(self.provider, "")
        self.api_key = settings.LLM_API_KEY

    # ------------------------------------------------------------------ #
    @staticmethod
    def prepare(system: str, prompt: str) -> tuple[str, str]:
        """Apply sensitive-column redaction (proposal Sec.19).

        Every prompt passes through here before it reaches any provider, so a
        new agent added later cannot bypass redaction by calling the client
        directly. Exposed separately so tests can assert on exactly what would
        be transmitted.
        """
        from app.core import redaction

        redactor = redaction.current()
        if not redactor.active:
            return system, prompt
        return (redactor.scrub(system) or "") + redactor.note(), redactor.scrub(prompt) or ""

    def complete(self, system: str, prompt: str) -> str:
        system, prompt = self.prepare(system, prompt)
        if self.provider == "mock":
            return _mock_complete(system, prompt)
        if not self.api_key:
            raise LLMError(
                f"LLM_PROVIDER is '{self.provider}' but LLM_API_KEY is empty in .env"
            )
        if self.provider == "anthropic":
            return self._anthropic(system, prompt)
        if self.provider == "openai":
            return self._openai(system, prompt)
        if self.provider == "gemini":
            return self._gemini(system, prompt)
        raise LLMError(f"Unknown LLM_PROVIDER: {self.provider}")

    def complete_json(self, system: str, prompt: str) -> dict | list:
        """Ask for JSON and parse it defensively (models like to add prose)."""
        raw = self.complete(
            system + "\n\nRespond with valid JSON only. No prose, no code fences.",
            prompt,
        )
        return _extract_json(raw)

    # ------------------------------------------------------------------ #
    def _anthropic(self, system: str, prompt: str) -> str:
        import httpx

        r = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": self.model,
                "max_tokens": settings.LLM_MAX_TOKENS,
                "system": system,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=120,
        )
        r.raise_for_status()
        blocks = r.json().get("content", [])
        return "".join(b.get("text", "") for b in blocks if b.get("type") == "text")

    def _openai(self, system: str, prompt: str) -> str:
        import httpx

        r = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "max_tokens": settings.LLM_MAX_TOKENS,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            },
            timeout=120,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    def _gemini(self, system: str, prompt: str) -> str:
        import httpx

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={self.api_key}"
        )
        r = httpx.post(
            url,
            json={
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"maxOutputTokens": settings.LLM_MAX_TOKENS},
            },
            timeout=120,
        )
        r.raise_for_status()
        parts = r.json()["candidates"][0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts)


# ---------------------------------------------------------------------- #
def _extract_json(text: str):
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # fall back to the first {...} or [...] block in the response
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = text.find(opener), text.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise LLMError(f"Could not parse JSON from LLM response: {text[:300]}")


def _mock_complete(system: str, prompt: str) -> str:
    """Deterministic stand-in so the pipeline runs without an API key.

    It reads what is being asked for and returns a structurally correct answer.
    The shapes match exactly what the real providers must return, so swapping in
    a real key changes quality, not plumbing.
    """
    p = prompt.lower()

    if "next step?" in p:
        # Single-agent baseline loop (app/evaluation/baselines.py). The mock
        # runs one tool then answers, which exercises the loop without
        # pretending to be analytical judgement.
        if "results so far" not in p:
            column = "region"
            for candidate in ("region", "product", "channel", "category"):
                if f"'{candidate}'" in p or f'"{candidate}"' in p:
                    column = candidate
                    break
            return json.dumps({
                "action": "tool", "tool": "run_dataframe_code",
                "params": {"operation": "value_counts", "params": {"column": column}},
            })
        return json.dumps({
            "action": "answer",
            "cause": "[mock provider: no analytical judgement available]",
            "confidence": "low",
            "evidence": "Mock provider - set a real LLM_PROVIDER for a meaningful baseline.",
        })

    if "generate hypotheses" in p or "possible explanations" in p:
        return json.dumps(
            [
                {
                    "statement": "The decline is concentrated in one segment rather than spread evenly.",
                    "variables": ["segment_column", "target_metric"],
                    "proposed_test": "groupby_aggregate on the segment column, then contribution analysis",
                },
                {
                    "statement": "The number of records (customers/orders) fell between periods.",
                    "variables": ["id_column", "date_column"],
                    "proposed_test": "count distinct ids per period and compare",
                },
                {
                    "statement": "The average value per record fell rather than the count.",
                    "variables": ["target_metric", "id_column"],
                    "proposed_test": "compare mean target per record across periods with a t-test",
                },
                {
                    "statement": "A supporting operational factor (availability, returns, price) changed.",
                    "variables": ["supporting_column", "target_metric"],
                    "proposed_test": "correlation between the supporting column and the target",
                },
            ]
        )

    if "plan" in p and "investigation" in p:
        return json.dumps(
            {
                "target_metric": "auto",
                "dimensions": [],
                "comparison_period": None,
                "steps": ["profile", "hypothesise", "test", "critique", "verify", "report"],
            }
        )

    if "critique" in p or "critic" in p:
        return json.dumps(
            {
                "concerns": [
                    "Seasonality has not been ruled out as an alternative explanation.",
                    "A correlation between two columns does not establish that one caused the other.",
                    "The comparison periods may differ in length or number of active records.",
                ],
                "verdict": "proceed_with_caveats",
            }
        )

    if "recommendation" in p:
        # echo back the finding so the mock produces a specific, readable action
        finding = ""
        for line in prompt.splitlines():
            if line.lower().startswith("verified finding:"):
                finding = line.split(":", 1)[1].strip().rstrip(".")
                break
        urgency = "high" if "declining" in p else "medium"
        return json.dumps(
            {
                "action": (
                    f"Investigate and act on: {finding}" if finding
                    else "Address the driver identified in the verified finding."
                ),
                "rationale": (
                    "This is where the measured movement is concentrated, so action here "
                    "addresses the largest share of the gap."
                ),
                "urgency": urgency,
            }
        )

    if "executive summary" in p or "report" in p:
        return (
            "This investigation examined the question against the supplied dataset. "
            "Findings below are computed from the data; each links to the exact "
            "calculation that produced it. Interpretations are marked separately "
            "from measurements, and unresolved hypotheses are listed as unresolved "
            "rather than treated as answered."
        )

    return "[mock LLM response]"


llm = LLMClient()
