from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.agents import chat as chat_agent
from app.config import settings
from app.core.deps import authorize_dataset, get_current_user, require_role
from app.database import get_db
from app.models import Conversation, Dataset, DatasetVersion, Message, User
from app.schemas.requests import ChatMessageRequest, ConversationCreate
from app.services import ingestion, profiling, rag

router = APIRouter(prefix="/api/chat", tags=["chat"])


def _serialise(message: Message) -> dict:
    return {
        "id": str(message.id),
        "role": message.role,
        "kind": message.kind,
        "content": message.content,
        "payload": message.payload,
        "created_at": message.created_at,
    }


def _conversation(db: Session, conversation_id: uuid.UUID, user: User) -> Conversation:
    conversation = db.get(Conversation, conversation_id)
    if not conversation:
        raise HTTPException(404, "Conversation not found")
    if conversation.owner_id and conversation.owner_id != user.id and user.role != "admin":
        raise HTTPException(403, "This conversation belongs to another user")
    return conversation


# --------------------------------------------------------------------- #
@router.post("/conversations", status_code=201)
def create_conversation(payload: ConversationCreate,
                        user: User = Depends(require_role("admin", "analyst")),
                        db: Session = Depends(get_db)):
    dataset_id = None
    if payload.dataset_id:
        dataset_id = authorize_dataset(db, uuid.UUID(payload.dataset_id), user).id

    conversation = Conversation(
        owner_id=user.id,
        title=payload.title or "New conversation",
        dataset_id=dataset_id,
    )
    db.add(conversation)
    db.flush()

    chat_agent.add_message(
        db, conversation, "assistant",
        chat_agent._greeting(db, conversation),
    )
    db.commit()
    return {"id": str(conversation.id), "title": conversation.title,
            "messages": [_serialise(m) for m in conversation.messages]}


@router.get("/conversations")
def list_conversations(user: User = Depends(get_current_user),
                       db: Session = Depends(get_db)):
    query = db.query(Conversation).filter(Conversation.is_archived.is_(False))
    if user.role != "admin":
        query = query.filter(Conversation.owner_id == user.id)
    rows = query.order_by(Conversation.updated_at.desc()).all()
    return {
        "count": len(rows),
        "conversations": [
            {
                "id": str(c.id),
                "title": c.title,
                "dataset_id": str(c.dataset_id) if c.dataset_id else None,
                "investigation_id": str(c.investigation_id) if c.investigation_id else None,
                "messages": len(c.messages),
                "updated_at": c.updated_at,
            }
            for c in rows
        ],
    }


@router.get("/conversations/{conversation_id}")
def get_conversation(conversation_id: uuid.UUID,
                     user: User = Depends(get_current_user),
                     db: Session = Depends(get_db)):
    """The whole thread. This is what makes a refresh lossless."""
    conversation = _conversation(db, conversation_id, user)
    dataset = db.get(Dataset, conversation.dataset_id) if conversation.dataset_id else None
    return {
        "id": str(conversation.id),
        "title": conversation.title,
        "dataset": {"id": str(dataset.id), "name": dataset.name} if dataset else None,
        "investigation_id": str(conversation.investigation_id)
        if conversation.investigation_id else None,
        "messages": [_serialise(m) for m in conversation.messages],
    }


@router.post("/conversations/{conversation_id}/messages")
def send_message(conversation_id: uuid.UUID, payload: ChatMessageRequest,
                 user: User = Depends(require_role("admin", "analyst")),
                 db: Session = Depends(get_db)):
    conversation = _conversation(db, conversation_id, user)
    if not payload.content.strip():
        raise HTTPException(422, "An empty message has nothing to answer")

    try:
        replies = chat_agent.respond(db, conversation, payload.content.strip())
        db.commit()
    except Exception as exc:  # noqa: BLE001
        db.rollback()
        raise HTTPException(500, f"The conversation failed: {exc}") from exc

    return {"messages": [_serialise(m) for m in replies]}


@router.post("/conversations/{conversation_id}/upload")
async def upload_in_chat(conversation_id: uuid.UUID, file: UploadFile = File(...),
                         user: User = Depends(require_role("admin", "analyst")),
                         db: Session = Depends(get_db)):
    """Attach data from inside the thread.

    A spreadsheet becomes the conversation's dataset. If the thread is waiting
    on a missing-evidence request, the file is merged into that investigation
    instead of replacing the dataset — otherwise answering a question with data
    would silently throw away the work already done.
    """
    conversation = _conversation(db, conversation_id, user)
    content = await file.read()
    if len(content) > settings.MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File exceeds the {settings.MAX_UPLOAD_MB} MB limit")

    extension = Path(file.filename).suffix.lower()

    # --- documents go to the knowledge base ---------------------------- #
    if extension in ingestion.DOCUMENT_EXTS:
        try:
            source = ingestion.register_file_source(db, file.filename, content,
                                                    owner_id=user.id)
            artifact = next(a for a in source.artifacts if a.artifact_type == "original")
            text = ingestion.extract_text(Path(artifact.storage_path))
            doc = rag.index_document(db, owner_id=user.id, title=file.filename, text=text,
                                     document_type="business_doc", source_id=source.id)
            source.status = "extracted"
            message = chat_agent.add_message(
                db, conversation, "assistant",
                f"Indexed {file.filename} as background context "
                f"({len(doc.chunks)} passages). I'll consult it when planning.",
                kind="dataset",
                payload={"document_id": str(doc.id), "name": file.filename},
            )
            db.commit()
            return {"messages": [_serialise(message)]}
        except Exception as exc:  # noqa: BLE001
            db.rollback()
            raise HTTPException(400, f"Could not read that document: {exc}") from exc

    if extension not in ingestion.TABULAR_EXTS:
        raise HTTPException(415, f"I can't read {extension} files")

    # --- answering an open evidence request ---------------------------- #
    investigation = None
    if conversation.investigation_id:
        from app.models import Investigation

        investigation = db.get(Investigation, conversation.investigation_id)
    pending = (
        chat_agent._open_request(db, investigation.id) if investigation else None
    )

    if pending:
        try:
            supplementary = ingestion.parse_dataframe(file.filename, content)
            dataset = db.get(Dataset, investigation.dataset_id)
            parent = db.get(DatasetVersion, investigation.version_id)
            version = ingestion.merge_supplementary(db, dataset, parent, supplementary)
            added = (version.cleaning_operations or {}).get("added_columns", [])
            investigation.version_id = version.id
            db.flush()

            chat_agent.add_message(
                db, conversation, "user",
                f"[attached {file.filename}]", kind="dataset",
                payload={"name": file.filename, "added_columns": added},
            )
            replies = chat_agent._resume(
                db, conversation, investigation, pending,
                f"Supplied file '{file.filename}' adds these columns: "
                + ", ".join(added),
            )
            db.commit()
            return {"messages": [_serialise(m) for m in replies]}
        except ValueError as exc:
            db.rollback()
            message = chat_agent.add_message(
                db, conversation, "assistant",
                f"I couldn't join that file to the data we're working with: {exc}",
                kind="error",
            )
            db.commit()
            return {"messages": [_serialise(message)]}

    # --- a new dataset for the thread ---------------------------------- #
    try:
        source = ingestion.register_file_source(db, file.filename, content,
                                                owner_id=user.id)
        dataset = ingestion.create_dataset_from_source(db, source)
        version = db.get(DatasetVersion, dataset.current_version_id)
        profile = profiling.profile_version(db, version, ingestion.load_version(version))
        source.status = "extracted"

        chat_agent.add_message(
            db, conversation, "user", f"[attached {file.filename}]",
            kind="dataset", payload={"name": file.filename},
        )
        message = chat_agent.attach_dataset(db, conversation, dataset, profile)
        db.commit()
    except Exception as exc:  # noqa: BLE001
        db.rollback()
        raise HTTPException(400, f"Could not read that file: {exc}") from exc

    return {"messages": [_serialise(message)]}


@router.delete("/conversations/{conversation_id}")
def archive_conversation(conversation_id: uuid.UUID,
                         user: User = Depends(require_role("admin", "analyst")),
                         db: Session = Depends(get_db)):
    conversation = _conversation(db, conversation_id, user)
    conversation.is_archived = True
    db.commit()
    return {"id": str(conversation.id), "archived": True}