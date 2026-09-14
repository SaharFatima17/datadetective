"""LLM access layer (proposal Sec.8).

The LLM is used ONLY for planning, hypothesis phrasing, critique and narrative.
It never produces a number that ends up in a finding - those come from the tool
layer. See Sec.3 ("trusted calculation") and Sec.13 ("grounding, not invention").

The `mock` provider lets the whole system run end-to-end with no API key, which
is what you use while building and testing everything below Phase 7.
"""

from __future__ import annotations

import json
import random
import re
import threading
import time

from app.config import settings

# Starting points only. Model names are retired on a schedule, so set LLM_MODEL
# in .env rather than relying on these — `scripts/list_models.py` prints the
# names your own key can currently use.
DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-4-5",
    "openai": "gpt-4o-mini",
    "gemini": "gemini-3.5-flash",
}


class LLMError(RuntimeError):
    pass


# Transient conditions worth retrying: rate limits, overload, gateway errors
# and read timeouts. A 400 or 401 is not retried — the request itself is wrong,
# and hammering the endpoint will not fix it.
RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


def _retry_after(response) -> float | None:
    """How long the provider says to wait, if it says at all.

    A 429 usually carries the answer. Guessing with exponential backoff when the
    server has told you the exact delay wastes quota: retry too early and the
    attempt is refused and counted, too late and a long run crawls.
    """
    header = response.headers.get("retry-after")
    if header:
        try:
            return float(header)
        except ValueError:
            pass

    # Google returns it inside the error body as e.g. {"retryDelay": "34s"}
    try:
        body = response.json()
    except Exception:  # noqa: BLE001
        return None
    for item in (body.get("error", {}) or {}).get("details", []) or []:
        delay = item.get("retryDelay")
        if isinstance(delay, str) and delay.endswith("s"):
            try:
                return float(delay[:-1])
            except ValueError:
                pass
    return None


def _is_daily_quota(response) -> bool:
    """Distinguish 'wait a minute' from 'come back tomorrow'.

    Retrying a per-day quota just burns the retry budget for nothing, and the
    run should stop with a message that says so.
    """
    text = response.text.lower()
    return "per day" in text or "perday" in text or "daily" in text


class _Pacer:
    """Keeps calls at least `LLM_MIN_INTERVAL_MS` apart.

    A free-tier key allows a fixed number of requests per minute. Without
    pacing an evaluation run fires as fast as it can, trips the limit within
    seconds, and then spends the rest of the run in backoff. Spacing the calls
    is faster overall than being throttled.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self) -> None:
        interval = settings.LLM_MIN_INTERVAL_MS / 1000
        if interval <= 0:
            return
        with self._lock:
            gap = time.monotonic() - self._last
            if gap < interval:
                time.sleep(interval - gap)
            self._last = time.monotonic()


_pacer = _Pacer()


class LLMClient:
    def __init__(self) -> None:
        self.provider = settings.LLM_PROVIDER.lower()
        self.model = settings.LLM_MODEL or DEFAULT_MODELS.get(self.provider, "")
        self.api_key = settings.LLM_API_KEY
        # Counted so an evaluation run can report how many calls it cost.
        self.calls = 0
        self.retries = 0

    def _with_retries(self, fn, system: str, prompt: str) -> str:
        """Call a provider, retrying transient failures with backoff.

        Without this a single rate-limit response part-way through an
        evaluation run discards every scenario already computed.
        """
        import httpx

        attempts = max(1, settings.LLM_MAX_RETRIES)
        last: Exception | None = None

        for attempt in range(attempts):
            _pacer.wait()
            try:
                self.calls += 1
                return fn(system, prompt)
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code

                if status == 429 and _is_daily_quota(exc.response):
                    raise LLMError(
                        f"{self.provider} daily quota is exhausted — retrying "
                        "will not help today. Either wait for the quota to "
                        "reset, use a key with a higher limit, or run fewer "
                        "scenarios with --scenarios."
                    ) from exc

                if status not in RETRYABLE_STATUS or attempt == attempts - 1:
                    detail = exc.response.text[:300]
                    if status == 404:
                        # Keep the provider's own words. An earlier version
                        # replaced them with a guess about retired models,
                        # which hid the real reason — a 404 here can also mean
                        # the endpoint or API version is wrong, not the name.
                        raise LLMError(
                            f"{self.provider} returned 404 for model "
                            f"'{self.model}'. The provider said: {detail}\n"
                            "Run `python scripts/check_llm.py` to test the "
                            "model directly and see the full response."
                        ) from exc
                    raise LLMError(
                        f"{self.provider} returned {status}: {detail}"
                    ) from exc
                last = exc
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                if attempt == attempts - 1:
                    raise LLMError(f"{self.provider} unreachable: {exc}") from exc
                last = exc

            # Prefer the provider's own figure; fall back to exponential
            # backoff with jitter so parallel callers do not retry in lockstep
            # and trip the limit again together.
            told = None
            if isinstance(last, httpx.HTTPStatusError):
                told = _retry_after(last.response)
            delay = told if told else min(30.0, (2 ** attempt) + random.uniform(0, 1))
            delay = min(delay + random.uniform(0, 1), 120.0)
            self.retries += 1
            time.sleep(delay)

        raise LLMError(f"{self.provider} failed after {attempts} attempts: {last}")

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
            return self._with_retries(self._anthropic, system, prompt)
        if self.provider == "openai":
            return self._with_retries(self._openai, system, prompt)
        if self.provider == "gemini":
            return self._with_retries(self._gemini, system, prompt)
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
        return _read_gemini(r.json(), self.model)


def _read_gemini(body: dict, model: str) -> str:
    """Pull the text out of a Gemini response, explaining an empty one.

    Gemini 3.x models reason before they answer, and that reasoning is charged
    against `maxOutputTokens`. When the budget runs out mid-thought the reply
    comes back 200 OK with `finishReason: MAX_TOKENS` and no `parts` at all.
    Indexing straight into `parts` raises a bare KeyError, which a caller then
    reports as "[error: 'parts']" — true, and useless. The real problem is a
    token budget, so say that.
    """
    candidates = body.get("candidates") or []
    if not candidates:
        blocked = (body.get("promptFeedback") or {}).get("blockReason")
        if blocked:
            raise LLMError(f"{model} refused the prompt: {blocked}")
        raise LLMError(f"{model} returned no candidates: {str(body)[:200]}")

    candidate = candidates[0]
    parts = (candidate.get("content") or {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts)
    if text.strip():
        return text

    reason = candidate.get("finishReason", "unknown")
    if reason == "MAX_TOKENS":
        raise LLMError(
            f"{model} used its entire {settings.LLM_MAX_TOKENS}-token budget "
            "before producing any output. Newer Gemini models spend tokens "
            "reasoning first, and that counts against the same budget. Raise "
            "LLM_MAX_TOKENS in .env (8000 is a safe starting point)."
        )
    if reason == "SAFETY":
        raise LLMError(f"{model} stopped on a safety filter.")
    raise LLMError(f"{model} returned an empty response (finishReason: {reason}).")


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