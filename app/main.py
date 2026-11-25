from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException
from psycopg import AsyncConnection, errors
from app.db import pool, get_connection
from app.cosine_similarity import cosine_similarity
from app.guard import check_mismatch_guard
from app.schema import ImageOut, PostCreate, PostOut, ReviewCreate, ReviewOut


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Open the connection pool at startup and close it at shutdown."""
    await pool.open()
    yield
    await pool.close()


app = FastAPI(lifespan=lifespan)


@app.get("/images", response_model=list[ImageOut])
async def list_images(conn: AsyncConnection = Depends(get_connection)):
    async with conn.cursor() as cur:
        await cur.execute(
            """
            SELECT id, category, attributes, caption, confidence, flagged, file_path
            FROM images
            ORDER BY id
            """
        )
        rows = await cur.fetchall()
    return [
        ImageOut(
            id=r[0],
            category=r[1],
            attributes=r[2],
            caption=r[3],
            confidence=r[4],
            flagged=r[5],
            file_path=r[6],
        )
        for r in rows
    ]


@app.get("/images/{image_id}", response_model=ImageOut)
async def get_image(image_id: int, conn: AsyncConnection = Depends(get_connection)):
    async with conn.cursor() as cur:
        await cur.execute(
            """
            SELECT id, category, attributes, caption, confidence, flagged, file_path
            FROM images
            WHERE id = %s
            """,
            (image_id,),
        )
        r = await cur.fetchone()
    if r is None:
        raise HTTPException(status_code=404, detail="Image not found")
    return ImageOut(
        id=r[0],
        category=r[1],
        attributes=r[2],
        caption=r[3],
        confidence=r[4],
        flagged=r[5],
        file_path=r[6],
    )


@app.get("/posts", response_model=list[PostOut])
async def list_posts(conn: AsyncConnection = Depends(get_connection)):
    async with conn.cursor() as cur:
        await cur.execute(
            "SELECT id, title, content, expected_category FROM posts ORDER BY id"
        )
        rows = await cur.fetchall()
    return [
        PostOut(id=r[0], title=r[1], content=r[2], expected_category=r[3])
        for r in rows
    ]


@app.get("/posts/{post_id}", response_model=PostOut)
async def get_post(post_id: int, conn: AsyncConnection = Depends(get_connection)):
    async with conn.cursor() as cur:
        await cur.execute(
            "SELECT id, title, content, expected_category FROM posts WHERE id = %s",
            (post_id,),
        )
        r = await cur.fetchone()
    if r is None:
        raise HTTPException(status_code=404, detail="Post not found")
    return PostOut(id=r[0], title=r[1], content=r[2], expected_category=r[3])


@app.post("/posts", response_model=PostOut, status_code=201)
async def create_post(
    payload: PostCreate, conn: AsyncConnection = Depends(get_connection)
):
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO posts (title, content, expected_category)
                VALUES (%s, %s, %s)
                RETURNING id
                """,
                (payload.title, payload.content, payload.expected_category),
            )
            post_id = (await cur.fetchone())[0]
        await conn.commit()
    except errors.UniqueViolation:
        await conn.rollback()
        raise HTTPException(status_code=409, detail="Post title already exists")

    return PostOut(id=post_id, **payload.model_dump())


async def _save_suggestion(
    conn: AsyncConnection,
    post_id: int,
    image_id: int,
    similarity_score: float,
    guard_passed: bool,
    rejection_reason: str | None,
) -> int:
    """Persist suggestion without burning SERIAL sequence IDs on updates."""
    async with conn.cursor() as cur:
        await cur.execute(
            "SELECT id FROM suggestions WHERE post_id = %s AND image_id = %s",
            (post_id, image_id),
        )
        existing = await cur.fetchone()

        if existing is not None:
            suggestion_id = existing[0]
            await cur.execute(
                """
                UPDATE suggestions
                SET similarity_score = %s,
                    guard_passed = %s,
                    rejection_reason = %s
                WHERE id = %s
                """,
                (similarity_score, guard_passed, rejection_reason, suggestion_id),
            )
        else:
            await cur.execute(
                """
                INSERT INTO suggestions (image_id, post_id, similarity_score, guard_passed, rejection_reason)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
                """,
                (image_id, post_id, similarity_score, guard_passed, rejection_reason),
            )
            suggestion_id = (await cur.fetchone())[0]

    await conn.commit()
    return suggestion_id


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
            suggestion_id = await _save_suggestion(
                conn, post_id, candidate["image_id"], candidate["similarity"], True, None
            )
            return {
                "suggestion_id": suggestion_id,
                "post_id": post_id,
                "result": "match",
                "suggestion": candidate,
            }

    top = scored[0]
    _, top_reason = check_mismatch_guard(
        top["category"],
        expected_category,
        top["similarity"],
        top["confidence"],
    )
    suggestion_id = await _save_suggestion(
        conn, post_id, top["image_id"], top["similarity"], False, top_reason
    )
    return {
        "suggestion_id": suggestion_id,
        "post_id": post_id,
        "result": "no confident match",
        "reason": top_reason,
        "top_candidate": top,
    }


@app.get("/reviews")
async def list_reviews(conn: AsyncConnection = Depends(get_connection)):
    """List all persisted suggestions along with guard outcomes and review decisions."""
    async with conn.cursor() as cur:
        await cur.execute(
            """
            SELECT s.id, s.post_id, p.title, s.image_id, i.file_path,
                   s.similarity_score, s.guard_passed, s.rejection_reason, rd.decision
            FROM suggestions s
            JOIN posts p ON p.id = s.post_id
            JOIN images i ON i.id = s.image_id
            LEFT JOIN review_decisions rd ON rd.suggestion_id = s.id
            ORDER BY s.id
            """
        )
        rows = await cur.fetchall()
    return [
        {
            "suggestion_id": r[0],
            "post_id": r[1],
            "post_title": r[2],
            "image_id": r[3],
            "file_path": r[4],
            "similarity_score": r[5],
            "guard_passed": r[6],
            "rejection_reason": r[7],
            "decision": r[8] or "pending",
        }
        for r in rows
    ]


@app.get("/reviews/{suggestion_id}")
async def inspect_suggestion(
    suggestion_id: int, conn: AsyncConnection = Depends(get_connection)
):
    """Inspect why a specific suggestion was selected or rejected."""
    async with conn.cursor() as cur:
        await cur.execute(
            """
            SELECT s.id, s.post_id, p.title, p.expected_category,
                   s.image_id, i.file_path, i.category, i.caption, i.confidence,
                   s.similarity_score, s.guard_passed, s.rejection_reason, rd.decision
            FROM suggestions s
            JOIN posts p ON p.id = s.post_id
            JOIN images i ON i.id = s.image_id
            LEFT JOIN review_decisions rd ON rd.suggestion_id = s.id
            WHERE s.id = %s
            """,
            (suggestion_id,),
        )
        r = await cur.fetchone()

    if r is None:
        raise HTTPException(status_code=404, detail="Suggestion not found")

    return {
        "suggestion_id": r[0],
        "post_id": r[1],
        "post_title": r[2],
        "expected_category": r[3],
        "image_id": r[4],
        "file_path": r[5],
        "image_category": r[6],
        "caption": r[7],
        "confidence": r[8],
        "similarity_score": r[9],
        "guard_passed": r[10],
        "rejection_reason": r[11],
        "decision": r[12] or "pending",
    }


@app.post("/reviews/{suggestion_id}", response_model=ReviewOut, status_code=201)
async def review_suggestion(
    suggestion_id: int,
    payload: ReviewCreate,
    conn: AsyncConnection = Depends(get_connection),
):
    """Approve or reject a suggestion (one review per suggestion)."""
    async with conn.cursor() as cur:
        await cur.execute("SELECT id FROM suggestions WHERE id = %s", (suggestion_id,))
        if await cur.fetchone() is None:
            raise HTTPException(status_code=404, detail="Suggestion not found")

    try:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO review_decisions (suggestion_id, decision)
                VALUES (%s, %s)
                RETURNING id
                """,
                (suggestion_id, payload.decision),
            )
            review_id = (await cur.fetchone())[0]
        await conn.commit()
    except errors.UniqueViolation:
        await conn.rollback()
        raise HTTPException(
            status_code=409, detail="Suggestion has already been reviewed"
        )

    return ReviewOut(
        id=review_id, suggestion_id=suggestion_id, decision=payload.decision
    )
