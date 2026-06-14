from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app
import logging

from app.config import settings
from app.database import init_db
from app.routers import prompts, evaluations, models, alerts

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting LLM Evaluation Platform...")
    await init_db()
    logger.info("Database initialized")
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title=settings.app_name,
    description="End-to-End LLM Evaluation & Monitoring Platform",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

app.include_router(prompts.router)
app.include_router(evaluations.router)
app.include_router(models.router)
app.include_router(alerts.router)


@app.get("/")
async def root():
    return {
        "app": settings.app_name,
        "version": "1.0.0",
        "docs": "/docs",
        "metrics": "/metrics",
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}
