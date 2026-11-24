from app.db import get_sync_connection, init_db
from app.embeddings import embed_text


def embed_images() -> None:
    """Embed captions for all images in PostgreSQL that do not yet have embeddings."""
    init_db()

    with get_sync_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT i.id, i.caption, i.file_path
                FROM images i
                LEFT JOIN image_embeddings ie ON ie.image_id = i.id
                WHERE ie.id IS NULL
                ORDER BY i.id
                """
            )
            rows = cur.fetchall()

            if not rows:
                print("All images already have embeddings.")
                return

            for image_id, caption, file_path in rows:
                vector = embed_text(caption)
                cur.execute(
                    "INSERT INTO image_embeddings (image_id, vector) VALUES (%s, %s)",
                    (image_id, vector),
                )
                print(f"Embedded image_id={image_id}: {file_path}")

        conn.commit()

    print(f"Done. Embedded {len(rows)} image(s).")


if __name__ == "__main__":
    embed_images()
