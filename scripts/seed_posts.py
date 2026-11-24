import json
from pathlib import Path
from app.db import get_sync_connection, init_db
from app.embeddings import embed_text
from app.schema import BlogPostSeed

POSTS_FILE = Path("data/blogs/posts.json")


def seed_posts() -> None:
    """Seed posts from data/blogs/posts.json and embed title + content into post_embeddings."""
    init_db()

    raw_posts = json.loads(POSTS_FILE.read_text(encoding="utf-8"))
    posts = [BlogPostSeed.model_validate(item) for item in raw_posts]

    embedded_count = 0
    skipped_count = 0

    with get_sync_connection() as conn:
        with conn.cursor() as cur:
            for post in posts:
                cur.execute(
                    """
                    INSERT INTO posts (title, content, expected_category)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (title) DO NOTHING
                    """,
                    (post.title, post.content, post.expected_category),
                )
                cur.execute("SELECT id FROM posts WHERE title = %s", (post.title,))
                post_id = cur.fetchone()[0]

                cur.execute(
                    "SELECT 1 FROM post_embeddings WHERE post_id = %s",
                    (post_id,),
                )
                if cur.fetchone() is not None:
                    skipped_count += 1
                    print(f"[SKIP] post_id={post_id}: {post.title}")
                    continue

                vector = embed_text(f"{post.title}\n\n{post.content}")
                cur.execute(
                    "INSERT INTO post_embeddings (post_id, vector) VALUES (%s, %s)",
                    (post_id, vector),
                )
                embedded_count += 1
                print(f"[EMBEDDED] post_id={post_id}: {post.title}")

        conn.commit()

    print(f"Done. Embedded: {embedded_count} | Skipped: {skipped_count}")


if __name__ == "__main__":
    seed_posts()
