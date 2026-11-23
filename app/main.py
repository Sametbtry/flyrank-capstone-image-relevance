from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.db import pool

app = FastAPI()


@app.get("/")
def root():
    return {
        "project_name" : "AI Image Understanding & Content Matching Engine",
    }

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Open the connection pool at startup and close it at shutdown."""
    await pool.open()
    yield
    await pool.close()


app = FastAPI(lifespan=lifespan)

@app.get("/health")
def get_health():
    return {"status": "ok"}