from fastapi import FastAPI

from api.routers import router

# from app.api.routes import router

app = FastAPI(title="Agentic RAG System")

app.include_router(router, prefix="/api")
@app.get("/health")
def health():
    return {"status": "ok"}