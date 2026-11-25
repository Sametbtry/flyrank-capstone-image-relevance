import json
from pathlib import Path
from app.cosine_similarity import cosine_similarity
from app.db import get_sync_connection
from app.guard import check_mismatch_guard

LABELS_FILE = Path("data/eval_labels.json")


def predict_for_post(post_vector: list[float], expected_category: str | None, images: list[tuple]) -> str | None:
    """Rank images by cosine similarity and return the first file_path that passes the guard."""
    scored = [
        (cosine_similarity(post_vector, img_vec), file_path, category, confidence)
        for file_path, category, confidence, img_vec in images
    ]
    scored.sort(key=lambda x: x[0], reverse=True)

    for similarity, file_path, category, confidence in scored[:5]:
        passed, _ = check_mismatch_guard(category, expected_category, similarity, confidence)
        if passed:
            return file_path
    return None


def evaluate() -> None:
    """Measure top-1 precision on the 15-post labeled evaluation set."""
    labels: dict[str, str | None] = json.loads(LABELS_FILE.read_text(encoding="utf-8"))

    with get_sync_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT p.id, p.title, p.expected_category, pe.vector
                FROM posts p
                JOIN post_embeddings pe ON pe.post_id = p.id
                ORDER BY p.id
                """
            )
            posts = {title: (post_id, exp_cat, vec) for post_id, title, exp_cat, vec in cur.fetchall()}

            cur.execute(
                """
                SELECT i.file_path, i.category, i.confidence, ie.vector
                FROM images i
                JOIN image_embeddings ie ON ie.image_id = i.id
                """
            )
            images = cur.fetchall()

    correct = 0
    total = len(labels)

    for title, expected_path in labels.items():
        if title not in posts:
            print(f"[MISSING POST] {title}")
            continue

        post_id, expected_category, post_vector = posts[title]
        predicted_path = predict_for_post(post_vector, expected_category, images)
        is_hit = predicted_path == expected_path
        if is_hit:
            correct += 1

        status = "PASS" if is_hit else "MISS"
        print(f"[{status}] post_id={post_id:02d} | predicted={predicted_path} | expected={expected_path}")

    precision = (correct / total) * 100 if total else 0.0
    print(f"\nTop-1 Precision: {correct}/{total} ({precision:.2f}%)")


if __name__ == "__main__":
    evaluate()
