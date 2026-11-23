import argparse
import time
from pathlib import Path
from app.db import get_sync_connection, init_db
from app.schema import Metadata
from app.vision import tag_image

IMAGES_DIR = Path("data/images")
MAX_RETRIES = 3
RATE_LIMIT_DELAY_SECONDS = 5
LOW_CONFIDENCE_THRESHOLD = 0.7
FALLBACK_MODELS = [
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
]


def tag_with_retries(image_path: Path) -> Metadata | None:
    """Try tag_image up to MAX_RETRIES times with exponential backoff and model fallback."""
    for attempt in range(1, MAX_RETRIES + 1):
        model = FALLBACK_MODELS[(attempt - 1) % len(FALLBACK_MODELS)]
        try:
            return tag_image(image_path, model=model)
        except Exception as e:
            if attempt == MAX_RETRIES:
                print(f"[FAIL] {image_path} failed after {MAX_RETRIES} attempts: {e}")
                return None
            wait = 2 ** attempt  # 2s, 4s
            print(
                f"[RETRY {attempt}/{MAX_RETRIES}] {image_path} ({model}) failed: {e} "
                f"-> retrying in {wait}s..."
            )
            time.sleep(wait)
    return None


def get_existing_file_paths(conn) -> set[str]:
    """Return the set of image file_paths already stored in PostgreSQL."""
    with conn.cursor() as cur:
        cur.execute("SELECT file_path FROM images")
        return {row[0] for row in cur.fetchall()}


def save_image_metadata(conn, file_path: str, metadata: Metadata) -> None:
    """Persist validated image metadata idempotently."""
    flagged = metadata.confidence < LOW_CONFIDENCE_THRESHOLD
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO images (category, attributes, caption, confidence, flagged, file_path)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (file_path) DO NOTHING
            """,
            (
                metadata.category,
                metadata.attributes,
                metadata.caption,
                metadata.confidence,
                flagged,
                file_path,
            ),
        )
    conn.commit()


def tag_dataset(limit: int | None = None) -> tuple[list[tuple[Path, Metadata]], list[Path], int]:
    """Tag untagged images in data/images and persist each result immediately to PostgreSQL."""
    init_db()

    image_paths = sorted(IMAGES_DIR.rglob("*.jpg"))
    if limit is not None:
        image_paths = image_paths[:limit]

    total = len(image_paths)
    results: list[tuple[Path, Metadata]] = []
    failures: list[Path] = []
    skipped = 0

    with get_sync_connection() as conn:
        existing_paths = get_existing_file_paths(conn)

        for idx, image_path in enumerate(image_paths, start=1):
            rel_path = image_path.as_posix()
            if rel_path in existing_paths:
                skipped += 1
                print(f"[{idx}/{total}] [SKIP] {rel_path} (already in DB)")
                continue

            metadata = tag_with_retries(image_path)
            if metadata is None:
                failures.append(image_path)
            else:
                save_image_metadata(conn, rel_path, metadata)
                existing_paths.add(rel_path)
                flagged = " [LOW CONFIDENCE - FLAGGED]" if metadata.confidence < LOW_CONFIDENCE_THRESHOLD else ""
                print(
                    f"[{idx}/{total}] [SAVED] {rel_path} -> "
                    f"category={metadata.category}, confidence={metadata.confidence:.2f}{flagged}"
                )
                results.append((image_path, metadata))

            if idx < total:
                time.sleep(RATE_LIMIT_DELAY_SECONDS)

        total_in_db = len(existing_paths)

    print(
        f"\nBatch complete — Tagged: {len(results)} | Skipped: {skipped} | "
        f"Failed: {len(failures)} | Total in DB: {total_in_db}"
    )
    return results, failures, skipped


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Idempotent batch image tagger using Gemini Vision + PostgreSQL.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images to inspect.")
    args = parser.parse_args()

    tag_dataset(limit=args.limit)
