from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import analytics, auth, chat, data, health, investigations
from app.config import settings

@asynccontextmanager
async def lifespan(_: FastAPI):
    Path(settings.STORAGE_DIR).mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(
    title=settings.APP_NAME,
    description="Autonomous Data Investigation Agent",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    # Authentication is a bearer token in a header, not a cookie, so
    # credentials are not needed — and with them enabled a wildcard origin is
    # rejected by browsers anyway.
    allow_origins=[o.strip() for o in settings.ALLOWED_ORIGINS.split(",") if o.strip()],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(auth.router)
app.include_router(data.router)
app.include_router(analytics.router)
app.include_router(investigations.router)
app.include_router(chat.router)

STATIC_DIR = Path(__file__).parent.parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def root():
    index = STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(index)
    return {"name": settings.APP_NAME, "docs": "/docs"}


@app.get("/api")
def api_info():
    return {
        "name": settings.APP_NAME,
        "version": "1.0.0",
        "auth_required": True,
        "llm_provider": settings.LLM_PROVIDER,
        "embedding_provider": settings.EMBEDDING_PROVIDER,
        "docs": "/docs",
    }