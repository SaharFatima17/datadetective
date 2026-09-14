"""Make one real call with your configured key and show exactly what comes back.

`list_models.py` says what the provider offers. This says whether a call
actually works, and prints the provider's own error verbatim when it doesn't —
no interpretation, because a guess about the cause is what sends you looking in
the wrong place.

    python scripts/check_llm.py

For Gemini it also tries the other API versions, since a model can be listed on
one version and served on another.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402

from app.config import settings  # noqa: E402
from app.llm.client import DEFAULT_MODELS  # noqa: E402

PROMPT = "Reply with the single word: ok"


def _show(label: str, response: httpx.Response) -> bool:
    ok = response.status_code == 200
    print(f"  {label:22s} {response.status_code}", end="")
    if ok:
        try:
            body = response.json()
            candidate = (body.get("candidates") or [{}])[0]
            parts = (candidate.get("content") or {}).get("parts") or []
            text = "".join(p.get("text", "") for p in parts).strip()
            if text:
                print(f"  -> {text[:60]}")
            else:
                # 200 with nothing in it is not success
                reason = candidate.get("finishReason", "unknown")
                print(f"  -> EMPTY (finishReason: {reason})")
                ok = False
        except Exception:  # noqa: BLE001
            print("  -> 200 but the body was not what we expected")
            ok = False
    else:
        print()
        detail = response.text.strip()
        for line in detail.splitlines()[:12]:
            print(f"      {line[:140]}")
    return ok


def check_gemini(model: str, key: str) -> None:
    payload = {
        "systemInstruction": {"parts": [{"text": "You are terse."}]},
        "contents": [{"parts": [{"text": PROMPT}]}],
        # Enough headroom for a model that reasons before answering. A tiny
        # budget returns 200 OK with an empty body, which reads as a failure
        # when the setup is in fact fine.
        "generationConfig": {"maxOutputTokens": 2000},
    }
    base = "https://generativelanguage.googleapis.com"
    working = []

    for version in ("v1beta", "v1", "v1alpha"):
        url = f"{base}/{version}/models/{model}:generateContent?key={key}"
        try:
            r = httpx.post(url, json=payload, timeout=60)
        except httpx.HTTPError as exc:
            print(f"  {version:22s} unreachable: {exc}")
            continue
        if _show(version, r):
            working.append(version)

    print()
    if working:
        print(f"Works on: {', '.join(working)}")
        if "v1beta" not in working:
            print(
                "The client uses v1beta. Since that failed, this model is served "
                "on a different version — pick a model that works on v1beta, or "
                "change the version in app/llm/client.py::_gemini."
            )
        else:
            print(f"Put this in .env:\n\n  LLM_MODEL={model}")
    else:
        print(
            "No API version served this model. Read the message above — it is "
            "the provider's own, not an interpretation. If it mentions quota or "
            "billing, the model exists but your key cannot use it; try a "
            "smaller one such as a flash-lite variant."
        )


def check_openai(model: str, key: str) -> None:
    r = httpx.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": model, "max_tokens": 20,
              "messages": [{"role": "user", "content": PROMPT}]},
        timeout=60,
    )
    _show("chat/completions", r)


def check_anthropic(model: str, key: str) -> None:
    r = httpx.post(
        "https://api.anthropic.com/v1/messages",
        headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                 "content-type": "application/json"},
        json={"model": model, "max_tokens": 20,
              "messages": [{"role": "user", "content": PROMPT}]},
        timeout=60,
    )
    _show("v1/messages", r)


def main() -> None:
    provider = settings.LLM_PROVIDER.lower()
    model = settings.LLM_MODEL or DEFAULT_MODELS.get(provider, "")
    key = settings.LLM_API_KEY

    print(f"provider  {provider}")
    print(f"model     {model or '(none set)'}")
    print(f"key       {'set, ' + str(len(key)) + ' chars' if key else 'EMPTY'}\n")

    if provider == "mock":
        print("LLM_PROVIDER is 'mock' — nothing to check.")
        return
    if not key:
        print("LLM_API_KEY is empty in .env.")
        return
    if not model:
        print("LLM_MODEL is empty and there is no default for this provider.")
        return

    if provider == "gemini":
        check_gemini(model, key)
    elif provider == "openai":
        check_openai(model, key)
    elif provider == "anthropic":
        check_anthropic(model, key)
    else:
        print(f"No check written for provider '{provider}'.")


if __name__ == "__main__":
    main()