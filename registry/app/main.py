from fastapi import FastAPI
from app.core.config import settings
from app.api.routes.health import router as health_router
from app.api.routes.models import router as models_router
from app.api.routes.artifacts import router as artifacts_router

app = FastAPI(title=settings.app_name)

app.include_router(health_router, tags=["health"])
app.include_router(models_router, prefix="/v1", tags=["registry"])
app.include_router(artifacts_router, prefix="/v1", tags=["artifacts"])