"""List the models your API key can actually use.

Providers retire model names on a schedule, and a retired name fails with a 404
that looks like a broken key. Rather than hard-coding a name that will expire,
ask the provider what it currently offers:

    python scripts/list_models.py

Put one of the printed names in .env as LLM_MODEL.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402

from app.config import settings  # noqa: E402


def gemini(key: str) -> list[str]:
    r = httpx.get(
        f"https://generativelanguage.googleapis.com/v1beta/models?key={key}",
        timeout=60,
    )
    r.raise_for_status()
    return [
        m["name"].removeprefix("models/")
        for m in r.json().get("models", [])
        if "generateContent" in m.get("supportedGenerationMethods", [])
    ]


def openai(key: str) -> list[str]:
    r = httpx.get("https://api.openai.com/v1/models",
                  headers={"Authorization": f"Bearer {key}"}, timeout=60)
    r.raise_for_status()
    return sorted(m["id"] for m in r.json().get("data", []))


def anthropic(key: str) -> list[str]:
    r = httpx.get("https://api.anthropic.com/v1/models",
                  headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
                  timeout=60)
    r.raise_for_status()
    return [m["id"] for m in r.json().get("data", [])]


FETCHERS = {"gemini": gemini, "openai": openai, "anthropic": anthropic}


def main() -> None:
    provider = settings.LLM_PROVIDER.lower()
    if provider == "mock":
        print("LLM_PROVIDER is 'mock'. Set a real provider in .env first.")
        return
    if provider not in FETCHERS:
        print(f"Don't know how to list models for provider '{provider}'.")
        return
    if not settings.LLM_API_KEY:
        print("LLM_API_KEY is empty in .env.")
        return

    try:
        names = FETCHERS[provider](settings.LLM_API_KEY)
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        hint = {
            401: "the key is rejected — check it was copied in full",
            403: "the key is valid but not permitted to list models",
            429: "rate limited; wait a minute and try again",
        }.get(status, "")
        print(f"{provider} returned {status}. {hint}")
        return

    if not names:
        print("The key works, but no models support text generation on it.")
        return

    print(f"{len(names)} model(s) available to this key:\n")
    for name in names:
        print(f"  {name}")

    _recommend(provider, names, settings.LLM_API_KEY)


def _version_key(name: str) -> tuple:
    """Sort model names newest-first by their version number.

    'gemini-2.5-flash' sorts below 'gemini-3.6-flash'. Plain string sorting
    gets this wrong, which is how an older model ended up recommended.
    """
    match = re.search(r"(\d+)(?:\.(\d+))?", name)
    major = int(match.group(1)) if match else 0
    minor = int(match.group(2)) if match and match.group(2) else 0
    return (major, minor)


def _probe(provider: str, model: str, key: str) -> bool:
    """Actually call the model. Being listed is not the same as being usable.

    Providers keep retired models in the catalogue but refuse them for new
    keys — 'no longer available to new users'. Only a real call settles it.
    """
    try:
        if provider == "gemini":
            r = httpx.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"{model}:generateContent?key={key}",
                json={"contents": [{"parts": [{"text": "say ok"}]}],
                      "generationConfig": {"maxOutputTokens": 5}},
                timeout=45)
        elif provider == "openai":
            r = httpx.post("https://api.openai.com/v1/chat/completions",
                           headers={"Authorization": f"Bearer {key}"},
                           json={"model": model, "max_tokens": 5,
                                 "messages": [{"role": "user", "content": "say ok"}]},
                           timeout=45)
        else:
            r = httpx.post("https://api.anthropic.com/v1/messages",
                           headers={"x-api-key": key,
                                    "anthropic-version": "2023-06-01",
                                    "content-type": "application/json"},
                           json={"model": model, "max_tokens": 5,
                                 "messages": [{"role": "user", "content": "say ok"}]},
                           timeout=45)
        return r.status_code == 200
    except httpx.HTTPError:
        return False


def _recommend(provider: str, names: list[str], key: str) -> None:
    """Find a small model that genuinely works, by trying it."""
    small = [n for n in names
             if any(t in n for t in ("flash", "mini", "haiku", "lite"))
             and not any(t in n for t in ("image", "tts", "audio", "transcribe",
                                          "video", "omni", "robotics",
                                          "computer-use", "preview"))]
    candidates = sorted(small, key=_version_key, reverse=True)[:4]
    if not candidates:
        return

    print("\nAn evaluation run makes hundreds of calls, so a small fast model "
          "is enough.\nBeing listed is not the same as being usable, so these "
          "are tested with a real call:\n")
    for model in candidates:
        works = _probe(provider, model, key)
        print(f"  {'works  ' if works else 'refused'}  {model}")
        if works:
            print(f"\nPut this in .env:\n\n  LLM_MODEL={model}")
            return
    print("\nNone of those worked. Run `python scripts/check_llm.py` after "
          "setting LLM_MODEL to see the provider's own message.")


if __name__ == "__main__":
    main()