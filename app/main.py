from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.database import connect_to_mongo, close_mongo_connection, db
from app.routers import suggestions, sprints, drafts, chat
from app.middleware.rate_limit import RequestSizeLimitMiddleware, check_llm_rate_limit
from app.jobs.scheduler import start_scheduler, stop_scheduler
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("gillo_ai")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Connect to MongoDB and ensure indexes exist
    logger.info(f"Starting {settings.SERVICE_NAME} in '{settings.ENVIRONMENT}' mode...")
    try:
        await connect_to_mongo()
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB during startup: {e}")

    # Start background scheduler if enabled
    start_scheduler()

    yield

    # Shutdown: Stop scheduler and close database connection pool
    stop_scheduler()
    await close_mongo_connection()


app = FastAPI(
    title="Gillo AI Service",
    description="System Architecture, ML Pipelines & AI Suggestions Inbox for Taskove / Gillo Agile Platform",
    version="1.0.0",
    lifespan=lifespan
)

# Request payload size guardrail
app.add_middleware(RequestSizeLimitMiddleware)

# Enable CORS with configured trusted origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["System"])
async def health_check():
    """Service health and database ping check."""
    mongo_status = "disconnected"
    if db.client:
        try:
            await db.client.admin.command("ping")
            mongo_status = "connected"
        except Exception as e:
            logger.error(f"MongoDB ping check failed: {e}")
            mongo_status = "unreachable"

    return {
        "status": "healthy",
        "service": settings.SERVICE_NAME,
        "environment": settings.ENVIRONMENT,
        "core_agile_mode": settings.CORE_AGILE_MODE,
        "database": mongo_status
    }


@app.get("/ai/models", tags=["System"])
async def get_active_models():
    """
    Exposes metadata, dataset variants, and test metrics for active models in the registry.
    Includes the mandatory note 'not validated on Taskove production data'.
    """
    from app.ml.registry import registry
    return {
        "service": settings.SERVICE_NAME,
        "models": registry.get_active_models_metadata()
    }


# Register feature routers with rate limits on LLM endpoints
app.include_router(suggestions.router)
app.include_router(sprints.router)
app.include_router(drafts.router, dependencies=[Depends(check_llm_rate_limit)])
app.include_router(chat.router, dependencies=[Depends(check_llm_rate_limit)])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=settings.SERVICE_PORT, reload=True)
