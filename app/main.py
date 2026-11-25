from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException
from psycopg import AsyncConnection
from app.db import pool, get_connection
from app.cosine_similarity import cosine_similarity
from app.guard import check_mismatch_guard


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Open the connection pool at startup and close it at shutdown."""
    await pool.open()
    yield
    await pool.close()


app = FastAPI(lifespan=lifespan)


@app.get("/")
def root():
    return {
        "project_name": "AI Image Understanding & Content Matching Engine",
    }


@app.get("/health")
def get_health():
    return {"status": "ok"}


@app.get("/suggestions/{post_id}/")
async def get_suggestions(
    post_id: int, conn: AsyncConnection = Depends(get_connection)
):
    async with conn.cursor() as cur:
        await cur.execute(
            """
            SELECT pe.vector, p.expected_category
            FROM post_embeddings pe
            JOIN posts p ON p.id = pe.post_id
            WHERE pe.post_id = %s
            """,
            (post_id,),
        )
        row = await cur.fetchone()
        if row is None:
            raise HTTPException(
                status_code=404, detail="Post not found or not embedded"
            )
        post_vector, expected_category = row

        await cur.execute(
            """
            SELECT i.id, i.file_path, i.caption, i.category, i.confidence, ie.vector
            FROM image_embeddings ie
            JOIN images i ON i.id = ie.image_id
            """
        )
        rows = await cur.fetchall()

    if not rows:
        raise HTTPException(status_code=404, detail="No embedded images found")

    scored = [
        {
            "image_id": r[0],
            "file_path": r[1],
            "caption": r[2],
            "category": r[3],
            "confidence": r[4],
            "similarity": cosine_similarity(post_vector, r[5]),
        }
        for r in rows
    ]
    scored.sort(key=lambda x: x["similarity"], reverse=True)

    for candidate in scored[:5]:
        passed, _ = check_mismatch_guard(
            candidate["category"],
            expected_category,
            candidate["similarity"],
            candidate["confidence"],
        )
        if passed:
            return {"post_id": post_id, "result": "match", "suggestion": candidate}

    top = scored[0]
    _, top_reason = check_mismatch_guard(
        top["category"],
        expected_category,
        top["similarity"],
        top["confidence"],
    )
    return {
        "post_id": post_id,
        "result": "no confident match",
        "reason": top_reason,
        "top_candidate": top,
    }
