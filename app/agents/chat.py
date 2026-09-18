"""The conversational agent (proposal Sec.17).

This does not re-implement the investigation. Every answer it gives comes from
the same orchestrator, the same tools and the same verification path the rest of
the system uses — the chat only decides what the person meant and renders the
result as a turn.

Intent is resolved by rules first, and only then refined by the model. That
order is deliberate:

  * the rules are deterministic, so the thread behaves identically in a demo
    with LLM_PROVIDER=mock as it does with a real key
  * a misrouted turn is worse than a slow one — sending "generate the report" to
    the hypothesis engine would start a second investigation instead of writing
    the report the person asked for

Anything the agent cannot classify becomes a clarifying question rather than a
guess, which is the same rule the investigation itself follows.
"""

from __future__ import annotations

import re
import uuid

from sqlalchemy.orm import Session

from app.agents import orchestrator
from app.llm.client import llm
from app.services import rag
from app.tools import registry
from app.models import (
    Conversation,
    Dataset,
    DatasetVersion,
    Finding,
    Investigation,
    Message,
    MissingEvidenceRequest,
    Report,
)

# ---------------------------------------------------------------- intents #
INTENTS = ("greeting", "farewell", "off_topic", "need_data", "investigate",
           "drill_down", "answer_request", "report", "explain", "about_system",
           "data_question", "capabilities", "acknowledge", "unresolved",
           "document_question", "unclear")

GREETING = re.compile(
    r"^\s*(hi+|hello|hey|salam|assalam|assalamu|aoa|greetings"
    r"|good (morning|evening|afternoon))\b", re.I)

FAREWELL = re.compile(
    r"^\s*(bye|goodbye|good night|khuda hafiz|allah hafiz|see you"
    r"|that'?s all|thats all|done for now)\b", re.I)

# Requests this assistant does not take. Listed explicitly so the refusal is the
# same with or without a model behind it — a demo running offline should not
# suddenly answer differently from one with a key.
OFF_TOPIC = (
    "joke", "poem", "story", "recipe", "song", "weather", "translate",
    "write me a", "write a python", "write code", "who won", "your opinion",
    "what do you think about", "capital of", "how old", "math problem",
)
THANKS = re.compile(r"\b(thanks|thank you|thx|ok|okay|got it|great|perfect|shukriya|theek)\b", re.I)
# "what does 123% mean" / "what do those numbers mean" — the words between
# "what does" and "mean" vary, so a keyword list cannot catch them all.
ASKS_MEANING = re.compile(r"\bwhat\s+(does|do|is|are)\b.{0,40}\bmean(s)?\b", re.I)

REPORT_WORDS = ("report", "write up", "write-up", "summarise", "summarize",
                "document", "pdf", "print")
EXPLAIN_WORDS = ("explain", "simple words", "simply", "plain english", "what does that mean",
                 "what do you mean", "elaborate", "clarify", "are you sure",
                 "how do you know", "prove", "why do you say", "so what",
                 "what does it mean", "in short")
CAPABILITY_WORDS = ("what can you do", "how do you work", "who are you",
                    "what do you do", "help me")
DRILL_WORDS = ("break", "split", " by ", " per ", "instead", "what about",
               "how about", "drill", "segment", "cut")
DATA_WORDS = ("how many", "how much", "what columns", "which columns",
              "date range", "how many rows", "what data", "what's in",
              "whats in", "describe the data", "quality issues")
CAUSE_WORDS = ("why", "cause", "reason", "drove", "driver", "responsible")
CHANGE_WORDS = ("decline", "drop", "fall", "fell", "decrease", "loss",
                "grew", "growth", "rise", "increase", "changed", "change")

# Definitions are written out rather than generated: a definition that drifts
# between runs is worse than no definition at all.
GLOSSARY = {
    "health score": "a single number out of 100 for how clean a dataset is. It starts at 100 and loses points for each quality issue, more for the serious ones. Below about 70 is worth cleaning before you rely on any total.",
    "driver": "a finding that explains the change: the movement is concentrated in one group, like a region or a product. Only driver findings can produce a recommendation, because only they point at something to act on.",
    "association": "two things move together. That is all it means. It does not establish that one caused the other, so it never produces a recommendation.",
    "measurement": "a statement of what changed, like 'revenue fell 23%'. It restates the question rather than answering it, so it cannot support an action.",
    "verified": "the calculation behind the finding was run a second time and the result matched exactly. Verification is mechanical here, not a model checking its own work.",
    "contribution": "a group's share of the total movement between two periods. It can exceed 100% when other groups moved the opposite way, which is itself a signal: the change is concentrated in one place while others offset it.",
    "mape": "mean absolute percentage error, which is how far a forecast was off on periods it was not trained on. It is always compared with a naive guess; a model that cannot beat 'next month equals last month' is not shown.",
    "confidence band": "the shaded range around a forecast. It is a 95% interval, not a margin of error on a single number, and it assumes current patterns continue.",
    "backtest": "fitting the model on older data and checking its predictions against periods that were held back. A forecast is not shown until it passes this.",
    "hypothesis": "a testable explanation proposed before looking at the answer. Each one ends as supported, rejected, or unresolved when the data cannot settle it.",
    "unresolved": "a hypothesis that could not be tested with the data available. It is listed rather than quietly dropped, so you can see what the answer does not cover.",
    "sensitive column": "a column whose values are kept out of anything sent to the model. It stays fully usable for grouping and statistics, because those run locally.",
    "checksum": "a fingerprint of a calculation's result. Re-running the calculation and comparing fingerprints is what 'verified' means here.",
    "lineage": "the chain of versions behind a dataset. Cleaning never overwrites; it writes a new version recording what changed.",
}


def _glossary_hit(text: str) -> str | None:
    lowered = text.lower()
    for term in sorted(GLOSSARY, key=len, reverse=True):
        if term in lowered:
            return term
    return None


def classify(text: str, *, has_dataset: bool, awaiting: bool, has_report: bool,
             columns: list[str], has_documents: bool = False) -> str:
    """Decide what the person is asking for.

    Order matters. An open request for data outranks everything, because the
    thread already asked a question and this reply belongs to it. Questions
    ABOUT an existing answer are checked before questions about the data, so
    "what does that mean?" is not mistaken for a new investigation — an earlier
    version made exactly that mistake and ran a fresh statistical investigation
    to answer a definition question.
    """
    lowered = text.strip().lower()
    words = lowered.split()

    if awaiting:
        return "answer_request"
    if FAREWELL.match(lowered) and len(words) <= 5:
        return "farewell"
    if GREETING.match(lowered) and len(words) <= 4:
        return "greeting"
    if any(w in lowered for w in OFF_TOPIC):
        return "off_topic"
    if THANKS.search(lowered) and len(words) <= 4:
        return "acknowledge"
    if any(w in lowered for w in CAPABILITY_WORDS):
        return "capabilities"
    if any(w in lowered for w in ("not establish", "could not test", "unresolved",
                                  "what did you miss", "what is missing",
                                  "rejected", "what could you not")):
        return "unresolved" if has_report else "unclear"
    if _glossary_hit(lowered):
        return "about_system"
    if ASKS_MEANING.search(lowered) or any(w in lowered for w in EXPLAIN_WORDS):
        if has_report:
            return "explain"
        return "need_data" if not has_dataset else "unclear"
    if any(w in lowered for w in REPORT_WORDS):
        if has_report:
            return "report"
        return "investigate" if has_dataset else "need_data"
    if not has_dataset:
        # Without a spreadsheet there is still the knowledge base. A question
        # about an indexed site or document is answerable from the documents
        # themselves, and refusing it because no CSV is attached would make
        # everything that was crawled unusable.
        return "document_question" if has_documents else "need_data"

    mentioned = [c for c in columns if c.lower() in lowered]
    if mentioned and has_report and any(w in lowered for w in DRILL_WORDS):
        return "drill_down"
    if any(w in lowered for w in DATA_WORDS):
        return "data_question"
    if any(w in lowered for w in CAUSE_WORDS) or any(w in lowered for w in CHANGE_WORDS):
        return "investigate"
    if lowered.endswith("?") and mentioned:
        return "drill_down" if has_report else "investigate"
    return "unclear"


def refine_with_model(text: str, rule_intent: str, context: dict) -> str:
    """Let a configured model correct the rules; ignore it when it is mock.

    The rules decide the path, so behaviour is identical with or without a key.
    A real model only helps with phrasings the keywords miss.
    """
    try:
        result = llm.complete_json(
            system=("Classify one message from a data-analysis conversation. "
                    'Reply with JSON {"intent": "..."} and nothing else. '
                    "Allowed: " + ", ".join(INTENTS) + ". "
                    "investigate = a NEW analytical question about the data. "
                    "drill_down = re-cut the existing answer by another column. "
                    "explain = asking about the answer already given. "
                    "about_system = asking what a term means. "
                    "data_question = asking about the dataset itself, not a cause."),
            prompt=(f"Message: {text}\n"
                    f"Data attached: {context['has_dataset']}\n"
                    f"Answer already given: {context['has_report']}\n"
                    f"Rule-based guess: {rule_intent}"),
        )
        intent = (result or {}).get("intent") if isinstance(result, dict) else None
        if intent in INTENTS:
            return intent
    except Exception:  # noqa: BLE001
        pass
    return rule_intent


# ------------------------------------------------------------ formatting #
def _finding_line(f: dict) -> str:
    return f"{f['statement']}"


def summarise_findings(report: dict) -> str:
    findings = report.get("findings") or []
    drivers = [f for f in findings if f.get("finding_type") == "driver"]
    others = [f for f in findings if f.get("finding_type") != "driver"]

    if not findings:
        return (
            "I could not establish anything from this data. Nothing in it "
            "explains the change, so I have not guessed at a cause. The "
            "unresolved hypotheses below say what was missing."
        )

    parts = []
    if drivers:
        parts.append(_finding_line(drivers[0]))
        parts.append(
            "That is the explanation — the change is concentrated there."
        )
    else:
        parts.append(
            "I found no single driver. What I can say is measured, not causal:"
        )
    for f in others[:2]:
        kind = "Correlation, not a cause" if f.get("finding_type") == "association" \
            else "A measurement, not an explanation"
        parts.append(f"{kind}: {f['statement']}")
    return "\n\n".join(parts)


def _open_request(db: Session, investigation_id) -> MissingEvidenceRequest | None:
    return (
        db.query(MissingEvidenceRequest)
        .filter(MissingEvidenceRequest.investigation_id == investigation_id,
                MissingEvidenceRequest.status == "open")
        .order_by(MissingEvidenceRequest.created_at)
        .first()
    )


def _profile(db: Session, conversation: Conversation) -> dict:
    """The stored profile. Answering from it means no recomputation."""
    if not conversation.dataset_id:
        return {}
    dataset = db.get(Dataset, conversation.dataset_id)
    version = db.get(DatasetVersion, dataset.current_version_id) if dataset else None
    return (version.profile_summary or {}) if version else {}


def _columns(profile: dict) -> list[str]:
    return [c["name"] for c in profile.get("columns", [])]


def _driver_finding(db: Session, investigation_id) -> Finding | None:
    if not investigation_id:
        return None
    return (
        db.query(Finding)
        .filter(Finding.investigation_id == investigation_id,
                Finding.finding_type == "driver")
        .order_by(Finding.created_at)
        .first()
    )


def _suggest(*items: str) -> dict:
    """Next steps offered alongside a reply — the guiding part of the chat."""
    return {"suggestions": [s for s in items if s]}


def _latest_report(db: Session, investigation_id) -> Report | None:
    if not investigation_id:
        return None
    return (
        db.query(Report)
        .filter(Report.investigation_id == investigation_id)
        .order_by(Report.version_number.desc())
        .first()
    )


TITLE_SYSTEM = (
    "You name conversations. Given the first thing a person asked, reply with a "
    "title of three to six words that says what the conversation is about. No "
    "quotes, no full stop, no preamble — the title only."
)

FILLER = {"why", "what", "which", "how", "when", "where", "did", "do", "does",
          "is", "are", "was", "were", "the", "a", "an", "our", "my", "we",
          "you", "please", "can", "could", "tell", "me", "about", "of", "in"}


def _fallback_title(text: str) -> str:
    """A readable title without a model.

    Filler words are dropped so "why did our revenue decline last quarter?"
    becomes "Revenue decline last quarter" rather than a truncated sentence.
    """
    words = [w.strip(" ?.,!\"'") for w in text.split()]
    kept = [w for w in words if w and w.lower() not in FILLER]
    if not kept:
        kept = [w for w in words if w][:5]
    title = " ".join(kept[:6])
    return (title[:1].upper() + title[1:])[:80] or "New conversation"


def title_for(text: str) -> str:
    """Name a thread from its first real message (proposal Sec.17)."""
    try:
        raw = llm.complete(system=TITLE_SYSTEM, prompt=text.strip()[:400])
        cleaned = raw.strip().splitlines()[0].strip().strip("\"'.")
        # a model that ignores the brief must not put a paragraph in the sidebar
        if 2 <= len(cleaned.split()) <= 10 and not cleaned.startswith("["):
            return cleaned[:80]
    except Exception:  # noqa: BLE001
        pass
    return _fallback_title(text)


def _provisional_title(db: Session, conversation: Conversation) -> str | None:
    """The placeholder taken from an attached file name, if that is the title."""
    if not conversation.dataset_id:
        return None
    dataset = db.get(Dataset, conversation.dataset_id)
    if not dataset:
        return None
    return _fallback_title(dataset.name.rsplit(".", 1)[0].replace("_", " "))


# Names that are only standing in until something better arrives. A thread
# should never sit in the sidebar as "New conversation" — but a greeting or a
# file name says nothing about what the person wanted, so a real question is
# allowed to replace either one. After that the name is left alone.
PLACEHOLDER_TITLES = {"New conversation", "Greeting"}


def maybe_name(db: Session, conversation: Conversation, text: str) -> None:
    """Name the thread from the first message, improving it on the first question.

    A thread opened from a report is already named after that investigation, so
    it is left alone — the name describes what the conversation is about, which
    is the point.
    """
    provisional = PLACEHOLDER_TITLES | {_provisional_title(db, conversation)}
    if conversation.title not in provisional:
        return

    stripped = text.strip()
    if GREETING.match(stripped) or len(stripped.split()) < 2:
        # Only if nothing better is there yet: a greeting must not overwrite a
        # title that came from an actual question.
        if conversation.title == "New conversation":
            conversation.title = "Greeting"
        return

    conversation.title = title_for(stripped)


def add_message(db: Session, conversation: Conversation, role: str, content: str,
                kind: str = "text", payload: dict | None = None) -> Message:
    message = Message(
        conversation_id=conversation.id,
        role=role,
        kind=kind,
        content=content,
        payload=payload,
    )
    db.add(message)
    db.flush()
    return message


# ------------------------------------------------------------------ turn #
def respond(db: Session, conversation: Conversation, text: str) -> list[Message]:
    """Handle one user message and return the assistant's reply messages."""
    add_message(db, conversation, "user", text)
    maybe_name(db, conversation, text)

    investigation = (db.get(Investigation, conversation.investigation_id)
                     if conversation.investigation_id else None)
    pending = _open_request(db, investigation.id) if investigation else None
    report = _latest_report(db, conversation.investigation_id)
    profile = _profile(db, conversation)
    columns = _columns(profile)

    from app.models import Document

    has_documents = db.query(Document.id).filter(
        (Document.owner_id == conversation.owner_id)
        | (Document.owner_id.is_(None))
    ).first() is not None

    context = {"has_dataset": conversation.dataset_id is not None,
               "has_report": report is not None}
    intent = classify(text, awaiting=pending is not None, columns=columns,
                      has_documents=has_documents, **context)
    if not pending:
        intent = refine_with_model(text, intent, context)

    # Greetings, thanks, capability questions and anything unclassified are
    # conversation rather than analysis. A model answers those in the person's
    # own register; without one the fixed wording below still applies.
    if intent in {"greeting", "farewell", "off_topic", "acknowledge",
                  "capabilities", "unclear"}:
        note = _state_note(db, conversation, report, has_documents)
        spoken = converse(db, conversation, text, note)
        if spoken:
            return [_say(db, conversation, spoken,
                         _next_steps(db, conversation, report))]

    if intent == "farewell":
        return [_say(db, conversation,
                     "Goodbye. Your investigations and documents stay here for "
                     "whenever you come back.")]
    if intent == "off_topic":
        return [_say(
            db, conversation,
            "That's outside what I do — I'm a data investigation agent. I can "
            "look into why a number moved in your data, explain a finding, or "
            "answer questions from the pages and documents you've indexed.",
            _next_steps(db, conversation, report))]
    if intent == "greeting":
        return [_say(db, conversation, _greeting(db, conversation, opening=False),
                     _next_steps(db, conversation, report))]
    if intent == "acknowledge":
        return [_acknowledge(db, conversation, report)]
    if intent == "capabilities":
        return [_capabilities(db, conversation)]
    if intent == "about_system":
        return [_define(db, conversation, text, report)]
    if intent == "need_data":
        return [_ask_for_data(db, conversation)]
    if intent == "answer_request":
        return _resume(db, conversation, investigation, pending, text)
    if intent == "report":
        return _deliver_report(db, conversation, report)
    if intent == "document_question":
        return [_answer_from_documents(db, conversation, text)]
    if intent == "unresolved":
        return [_unresolved(db, conversation, report)]
    if intent == "explain":
        return [_explain(db, conversation, investigation)]
    if intent == "data_question":
        return [_describe_data(db, conversation, profile)]
    if intent == "drill_down":
        return [_drill(db, conversation, text, columns, investigation)]
    if intent == "investigate":
        return _investigate(db, conversation, text)
    return [_unclear(db, conversation, report, columns)]


def _say(db, conversation, content, payload=None, kind="text"):
    return add_message(db, conversation, "assistant", content, kind=kind,
                       payload=payload)


def _next_steps(db: Session, conversation: Conversation, report) -> dict:
    if not conversation.dataset_id:
        return _suggest("What can you do?")
    if not report:
        return _suggest("Why did revenue decline?", "What's in this data?")
    return _suggest("Explain that in simple words", "Write the report",
                    "Break it down by another column")


def opening_for_investigation(db: Session, investigation) -> str:
    """The first message when a thread is opened from a finished report.

    Someone reading this did not run the investigation — they were handed the
    result. So the opening states what was found and, just as importantly, what
    the finding does not claim, before they ask anything.
    """
    driver = _driver_finding(db, investigation.id)
    report = _latest_report(db, investigation.id)

    lines = [f'This is about the investigation "{investigation.question}".']

    if driver:
        lines.append(driver.statement)
        lines.append(
            "That is the explanation: the change is concentrated there rather "
            "than spread evenly. Every figure in it came from a calculation you "
            "can make me run again."
        )
    else:
        lines.append(
            "No single cause was established. The change was measured, but "
            "nothing in the data accounted for it, so none was named."
        )

    unresolved = ((report.content or {}).get("unresolved_hypotheses")
                  if report else None)
    if unresolved:
        lines.append(
            f"{len(unresolved)} possible explanation"
            f"{'s' if len(unresolved) != 1 else ''} could not be tested with the "
            "data available. Ask what they were if that matters."
        )

    lines.append("What would you like me to go through?")
    return "\n\n".join(lines)


CONVERSE_SYSTEM = """You are DataDetective's assistant. You investigate business
data. Talk like a capable colleague: warm, brief, plain English — two or three
sentences unless more is genuinely needed.

What this system does: someone attaches a spreadsheet and asks a question in
ordinary language. Agents form hypotheses, test each one with pandas, SQL and
statistical tests, verify every number by re-running the calculation that
produced it, and report what could not be established. Documents and websites
can also be indexed, and questions about those are answered from the passages
themselves.

YOU MUST NOT STATE A FIGURE about the person's data — no percentage, no trend,
no finding. You have not seen their data and you cannot calculate. If they ask
something answerable from their data, say you will look and let the analysis
run. Inventing a number here would destroy the only thing this system is for.

STAY ON YOUR SUBJECT. You are not a general assistant. Greetings, thanks and
goodbyes get a natural reply. Everything else must concern data investigation,
this system, or the documents that have been indexed.

If someone asks for a joke, a poem, a recipe, general knowledge, coding help, or
anything else outside data investigation, say plainly in one sentence that this
is not what you do, and offer what you can: investigate a question about their
data, explain a finding, or answer from the indexed documents. Do not apologise
at length and do not comply partially — a joke told "just this once" teaches the
person to expect a general chatbot, and then to trust its numbers too."""


def converse(db: Session, conversation: Conversation, text: str,
             context_note: str) -> str | None:
    """Reply in the person's own register, when a model is configured.

    The rule-based replies elsewhere are exact but fixed: the same sentence
    every time, whatever was asked. For greetings, thanks, and anything that
    does not map onto an analytical action, that reads like a switchboard. This
    hands those turns to the model with the thread so far, and returns None when
    no model is configured so the caller falls back to the fixed wording.

    The model is given the state of the conversation but never the data, and is
    told plainly that it must not produce figures. Everything numerical still
    comes from the tool layer.
    """
    already_greeted = (
        db.query(Message)
        .filter(Message.conversation_id == conversation.id,
                Message.role == "assistant")
        .count()
        > 0
    )

    history = (
        db.query(Message)
        .filter(Message.conversation_id == conversation.id)
        .order_by(Message.created_at.desc())
        .limit(8)
        .all()
    )
    transcript = "\n".join(
        f"{m.role}: {m.content[:400]}" for m in reversed(history)
    )

    try:
        reply = llm.complete(
            system=CONVERSE_SYSTEM,
            prompt=(f"Current state: {context_note}\n\n"
                    f"Conversation so far:\n{transcript}\n\n"
                    + ("You have already greeted this person in this thread. Do "
                       "not greet again — acknowledge briefly and move on to "
                       "what they can do next.\n\n" if already_greeted else "")
                    + "Reply to the last user message."),
        )
    except Exception:  # noqa: BLE001
        return None

    reply = (reply or "").strip()

    # Anything that is not prose is not an answer. The offline stand-in returns
    # a placeholder in brackets, and — because it matches on prompt keywords —
    # can return a JSON analysis plan instead. Either one shown to the person
    # would be the machinery leaking through the conversation, so both are
    # rejected and the caller falls back to its fixed wording.
    if (not reply
            or len(reply) < 3
            or reply[0] in "[{"
            or reply.startswith("```")
            or '"target_metric"' in reply
            or "mock LLM" in reply):
        return None
    return reply


def _state_note(db: Session, conversation: Conversation, report, has_documents: bool) -> str:
    """One line describing what is loaded, so replies suggest the right next step."""
    parts = []
    if conversation.dataset_id:
        dataset = db.get(Dataset, conversation.dataset_id)
        parts.append(f"a spreadsheet is attached ({dataset.name if dataset else 'unknown'})")
    else:
        parts.append("no spreadsheet is attached")
    if report:
        driver = _driver_finding(db, conversation.investigation_id)
        parts.append("an investigation has finished"
                     + (f"; its explanation was: {driver.statement}" if driver else
                        " and found no explanation"))
    if has_documents:
        parts.append("documents and pages are indexed and can be asked about")
    return "; ".join(parts) + "."


def _acknowledge(db, conversation, report):
    if report:
        return _say(db, conversation,
                    "Happy to help. Anything else you'd like to look at?",
                    _suggest("Write the report", "Break it down by another column",
                             "Ask a different question"))
    return _say(db, conversation, "Any time. What would you like to look at?",
                _next_steps(db, conversation, report))


def _capabilities(db, conversation):
    return _say(
        db, conversation,
        "I investigate a question against your data and show my working.\n\n"
        "Attach a spreadsheet and ask something like \"why did revenue decline?\". "
        "I form hypotheses, test each one with SQL and statistics, re-run every "
        "number to check it, and tell you which hypotheses I could not settle.\n\n"
        "I don't invent figures. Every number comes from a calculation you can "
        "make me run again in front of you. If the data can't answer the question, "
        "I ask you for what's missing instead of guessing.",
        _suggest("What's in this data?", "Why did revenue decline?"),
    )


def _define(db, conversation, text, report):
    """Answer a terminology question from the glossary — nothing is computed."""
    term = _glossary_hit(text)
    return _say(db, conversation, f"{term.capitalize()} is {GLOSSARY[term]}",
                _next_steps(db, conversation, report))


def _describe_data(db: Session, conversation: Conversation, profile: dict):
    """Describe the dataset from the stored profile, without a new run."""
    if not profile:
        return _ask_for_data(db, conversation)

    columns = profile.get("columns", [])
    by_type: dict[str, list[str]] = {}
    for c in columns:
        by_type.setdefault(c.get("inferred_type", "other"), []).append(c["name"])

    issues = profile.get("issue_counts", {}) or {}
    lines = [f"{profile.get('row_count', 0):,} rows across {len(columns)} columns, "
             f"health score {profile.get('health_score')}/100."]
    for kind in ("datetime", "numeric", "categorical", "id", "text"):
        if by_type.get(kind):
            lines.append(f"{kind.capitalize()}: {', '.join(by_type[kind])}")
    total = sum(issues.values()) if issues else 0
    if total:
        lines.append(f"{total} quality issues detected ({issues.get('high', 0)} high "
                     f"severity). The Clean tab lists them and what it proposes to do.")

    return _say(db, conversation, "\n\n".join(lines),
                _suggest("Why did revenue decline?", "What is a health score?"))


def _explain(db: Session, conversation: Conversation, investigation):
    """Reword the answer already established. Nothing is recomputed."""
    driver = _driver_finding(db, investigation.id) if investigation else None
    if not driver:
        return _say(
            db, conversation,
            "There was no single explanation to give. I measured the change, but "
            "nothing in the data accounted for it, so I stopped rather than naming "
            "a cause I could not support.",
            _suggest("What's in this data?", "Ask a different question"))

    unit = driver.unit or "the metric"
    content = (
        f"In plain terms: {driver.statement}\n\n"
        f"What that means is the movement in {unit} is not spread evenly. It is "
        "concentrated in one group, so acting there addresses most of the gap "
        "rather than a slice of it.\n\n"
        f"How I know: {driver.evidence_summary}\n\n"
        f"What it does not say: {driver.caveats or 'nothing further.'}"
    )
    return _say(db, conversation, content,
                _suggest("Write the report", "Break it down by another column"))


def _drill(db: Session, conversation: Conversation, text: str,
           columns: list[str], investigation):
    """Re-cut the same question by another column.

    One tool call through the normal registry — logged, checksummed and
    re-runnable from the Evidence tab — rather than a second investigation.
    """
    mentioned = [c for c in columns if c.lower() in text.lower()]
    plan = (investigation.plan or {}) if investigation else {}
    metric = plan.get("target_metric")
    date_column = plan.get("date_column")

    if not (mentioned and metric and date_column and investigation):
        return _unclear(db, conversation, None, columns)

    column = mentioned[0]
    try:
        result, run = registry.call_tool(
            db, "run_dataframe_code",
            {"dataset_id": str(conversation.dataset_id),
             "operation": "period_contribution",
             "params": {"date_column": date_column, "metric": metric, "by": column},
             "version_id": str(investigation.version_id)},
            investigation_id=investigation.id, agent_name="chat")
    except Exception as exc:  # noqa: BLE001
        return _say(db, conversation,
                    f"I couldn't break {metric} down by {column}: {exc}", kind="error")

    rows = (result or {}).get("result") or []
    if run.status != "success" or not rows:
        return _say(db, conversation,
                    f"I couldn't break {metric} down by {column}: "
                    f"{run.error_message or 'no groups were produced'}.", kind="error")

    top = rows[0]
    lines = [f"{metric} by {column}, over the same two periods:",
             f"{top['group']} carries {top['contribution_to_total_change_pct']}% of the "
             f"{result.get('direction', 'change')} "
             f"({top['previous_per_period']} to {top['current_per_period']} per period, "
             f"{top['change_pct']}%)."]
    rest = ", ".join(f"{r['group']} {r['change_pct']:+.1f}%" for r in rows[1:4]
                     if r.get("change_pct") is not None)
    if rest:
        lines.append(f"The rest: {rest}.")
    lines.append("This is the same calculation the investigation uses, run on a "
                 "different column. It is logged, so you can re-run it from the "
                 "Evidence tab.")

    return _say(db, conversation, "\n\n".join(lines), kind="findings",
                payload={"investigation_id": str(investigation.id),
                         "breakdown": result, "tool_run_id": str(run.id),
                         **_suggest("Write the report",
                                    "Explain that in simple words")})


def _answer_from_documents(db: Session, conversation: Conversation, text: str):
    """Answer from the knowledge base when there is no dataset in play.

    The sources are listed with the answer, always. A document answer cannot be
    re-computed the way a finding can, so the only thing that makes it checkable
    is showing where each part came from.
    """
    result = rag.answer_from_documents(
        db, text, top_k=5, owner_id=conversation.owner_id)

    if not result["answered"]:
        # Not every message with no spreadsheet attached is a question for the
        # documents. "Tell me a joke" reaching the knowledge base and being told
        # nothing is indexed about it is a switchboard answering a person.
        spoken = converse(db, conversation, text,
                          _state_note(db, conversation, None, True))
        if spoken:
            return _say(db, conversation, spoken,
                        _suggest("Attach a spreadsheet", "What can you do?"))
        return _say(db, conversation, result["answer"],
                    _suggest("What can you do?", "Attach a spreadsheet"))

    lines = [result["answer"]]
    if not result.get("composed"):
        for s_ in result["sources"][:3]:
            lines.append(f"[{s_['n']}] {s_['title']}\n{s_['excerpt'][:300]}")
    lines.append(
        "This comes from indexed documents, not from a calculation. It can be "
        "traced to the passage above; it cannot be re-computed the way a finding "
        "from a spreadsheet can."
    )

    return _say(db, conversation, "\n\n".join(lines), kind="findings",
                payload={"sources": result["sources"],
                         **_suggest("Ask something else", "Attach a spreadsheet")})


def _unresolved(db, conversation, report):
    """What the investigation could not settle.

    Reported as plainly as the findings. An answer that hides its gaps is the
    kind of answer this system exists to avoid.
    """
    content = (report.content or {}) if report else {}
    unresolved = content.get("unresolved_hypotheses") or []
    if not unresolved:
        return _say(db, conversation,
                    "Every hypothesis raised was either supported or rejected by "
                    "the data — none was left unresolved.",
                    _suggest("Explain that in simple words", "Write the report"))

    lines = ["These were raised and could not be settled with the data available:"]
    lines.extend(f"- {h}" for h in unresolved)
    lines.append(
        "They are listed rather than dropped, so the answer's limits are visible. "
        "Supplying data that covers them would let me test them."
    )
    return _say(db, conversation, "\n\n".join(lines),
                _suggest("Explain that in simple words", "Write the report"))


def _unclear(db, conversation, report, columns):
    hint = f"\n\nColumns I can work with: {', '.join(columns[:5])}." if columns else ""
    return _say(
        db, conversation,
        "I'm not sure what you're asking for. I can investigate a change in your "
        "data, explain an answer I've already given, define a term I've used, or "
        "write the report." + hint,
        _next_steps(db, conversation, report))


def _greeting(db: Session, conversation: Conversation, opening: bool = True) -> str:
    """The opening line, and a shorter one for a greeting mid-thread.

    Repeating the full introduction when someone says hello makes the thread
    look like it has forgotten the conversation so far.
    """
    dataset = db.get(Dataset, conversation.dataset_id) if conversation.dataset_id else None

    if dataset:
        return (
            f"We're working with {dataset.name}. What would you like to find out?"
            if opening
            else f"Still here, with {dataset.name} loaded. What would you like to know?"
        )

    if opening:
        return (
            "Hello. Attach a spreadsheet and tell me what you want to know — "
            "something like \"why did revenue decline?\". I'll form hypotheses, "
            "test them against your data, and tell you what I could not "
            "establish."
        )
    return "Still here — attach a spreadsheet whenever you're ready."


def _ask_for_data(db: Session, conversation: Conversation) -> Message:
    return add_message(
        db, conversation, "assistant",
        "I don't have any data in this conversation yet. Attach a spreadsheet "
        "(CSV or Excel) using the paperclip, and then ask your question.",
        kind="request",
        payload={"needs": "dataset"},
    )


def _investigate(db: Session, conversation: Conversation, question: str) -> list[Message]:
    dataset = db.get(Dataset, conversation.dataset_id)
    if not dataset:
        return [_ask_for_data(db, conversation)]

    try:
        investigation = orchestrator.start_investigation(
            db, dataset.id, question, owner_id=conversation.owner_id
        )
        state = orchestrator.run(db, investigation)
    except Exception as exc:  # noqa: BLE001
        return [
            add_message(
                db, conversation, "assistant",
                f"I couldn't complete that investigation: {exc}",
                kind="error",
            )
        ]

    conversation.investigation_id = investigation.id
    db.flush()

    report = state.get("report") or {}
    out = [
        add_message(
            db, conversation, "assistant",
            summarise_findings(report),
            kind="findings",
            payload={
                "investigation_id": state["investigation_id"],
                "findings": report.get("findings", []),
                "forecasts": report.get("forecasts", []),
                "recommendations": report.get("recommendations", []),
                "unresolved": report.get("unresolved_hypotheses", []),
                **_suggest("Explain that in simple words", "Write the report",
                           "Break it down by another column"),
            },
        )
    ]

    # If the run paused for evidence, ask for it as its own turn — a request
    # buried under findings gets missed.
    for request in state.get("open_requests", []):
        out.append(
            add_message(
                db, conversation, "assistant", request["question"],
                kind="request",
                payload={"needs": "evidence", "request_id": request["id"],
                         "reason": request.get("reason")},
            )
        )
    return out


def _resume(db: Session, conversation: Conversation, investigation: Investigation,
            request: MissingEvidenceRequest, text: str) -> list[Message]:
    try:
        state = orchestrator.resume(db, investigation, request.id, text)
    except Exception as exc:  # noqa: BLE001
        return [
            add_message(db, conversation, "assistant",
                        f"I couldn't use that answer: {exc}", kind="error")
        ]

    report = state.get("report") or {}
    out = [
        add_message(
            db, conversation, "assistant",
            "Thanks — I've picked the investigation back up.\n\n"
            + summarise_findings(report),
            kind="findings",
            payload={
                "investigation_id": state["investigation_id"],
                "findings": report.get("findings", []),
                "forecasts": report.get("forecasts", []),
                "recommendations": report.get("recommendations", []),
                "unresolved": report.get("unresolved_hypotheses", []),
                **_suggest("Explain that in simple words", "Write the report",
                           "Break it down by another column"),
            },
        )
    ]
    for pending in state.get("open_requests", []):
        out.append(
            add_message(db, conversation, "assistant", pending["question"],
                        kind="request",
                        payload={"needs": "evidence", "request_id": pending["id"],
                                 "reason": pending.get("reason")})
        )
    return out


def _deliver_report(db: Session, conversation: Conversation,
                    report: Report | None) -> list[Message]:
    if not report:
        return [
            add_message(
                db, conversation, "assistant",
                "There's nothing to write up yet — ask me a question about your "
                "data first, and I'll report on what I find.",
            )
        ]
    return [
        add_message(
            db, conversation, "assistant",
            report.executive_summary or "Here is the report.",
            kind="report",
            payload={
                "investigation_id": str(report.investigation_id),
                "report_id": str(report.id),
                "version": report.version_number,
                **(report.content or {}),
            },
        )
    ]


def attach_dataset(db: Session, conversation: Conversation, dataset: Dataset,
                   profile: dict | None = None) -> Message:
    """Called when a file is uploaded inside the thread."""
    conversation.dataset_id = dataset.id
    conversation.investigation_id = None
    if conversation.title == "New conversation":
        conversation.title = _fallback_title(
            dataset.name.rsplit(".", 1)[0].replace("_", " ")
        )
    db.flush()

    health = (profile or {}).get("health_score")
    rows = (profile or {}).get("row_count")
    issues = (profile or {}).get("issue_counts") or {}
    high = issues.get("high", 0)

    lines = [f"Got it — {dataset.name}."]
    if rows:
        lines.append(f"{rows:,} rows, health score {health}/100.")
    if high:
        lines.append(
            f"{high} high-severity data quality issue"
            f"{'s' if high != 1 else ''} — worth cleaning before you rely on "
            "the numbers."
        )
    lines.append("What would you like to find out?")

    return add_message(
        db, conversation, "assistant", " ".join(lines),
        kind="dataset",
        payload={"dataset_id": str(dataset.id), "name": dataset.name,
                 "profile": profile},
    )