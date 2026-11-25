# AI Image Understanding & Content Matching Engine

An AI backend service that inspects a 50-image wildlife library with **Gemini Vision**, validates structured metadata with **Pydantic**, embeds image captions and blog posts into a shared semantic vector space (`sentence-transformers` `all-MiniLM-L6-v2`), ranks candidates by **cosine similarity**, and filters every recommendation through a three-check **Mismatch Guard** before surfacing it to a human review API.

See [DESIGN.md](DESIGN.md) for the full design document, database schema, guard rules, and dataset breakdown.

---

## Architecture

```text
Images (data/images/*) ──(scripts/batch_image.py)──> Gemini Vision API
                                                          │
                                        Pydantic Metadata + Cost Logging
                                                          ▼
                                               PostgreSQL: images
                                                          │
                                             (scripts/embed_images.py)
                                             all-MiniLM-L6-v2 (384-d)
                                                          ▼
                                          PostgreSQL: image_embeddings
                                                          │
Posts (data/blogs/posts.json) ──(scripts/seed_posts.py)───┼──> PostgreSQL: posts + post_embeddings
                                                          │
GET /suggestions/{post_id}/ ──────────────────────────────┘
  ├── 1. Cosine Similarity Ranking (post_vector × image_vectors, top 5)
  ├── 2. Mismatch Guard (app/guard.py)
  │       ├── Check 1: Subject Match (image.category == post.expected_category)
  │       ├── Check 2: Similarity Threshold (similarity >= 0.33)
  │       └── Check 3: Confidence Threshold (confidence >= 0.70)
  ├── 3. Persist outcome to PostgreSQL: suggestions
  │       ├── Match found    -> {"result": "match", "suggestion": ...}
  │       └── All rejected   -> {"result": "no confident match", "reason": ...}
  └── 4. Human Review API: GET/POST /reviews/{suggestion_id} -> PostgreSQL: review_decisions
```

```mermaid
flowchart TD
    subgraph Ingestion["1. Batch Ingestion & Embeddings"]
        IMG["data/images/ (50 wildlife photos)"] -->|scripts/batch_image.py| VIS["Gemini Vision (app/vision.py)\nPydantic Metadata + Cost Log"]
        VIS -->|idempotent upsert| DB_IMG[("PostgreSQL: images")]
        DB_IMG -->|scripts/embed_images.py| EMB["Local Embeddings (app/embeddings.py)\nall-MiniLM-L6-v2"]
        EMB --> DB_IEMB[("PostgreSQL: image_embeddings")]

        POSTS["data/blogs/posts.json (15 posts)"] -->|scripts/seed_posts.py| DB_POSTS[("PostgreSQL: posts")]
        DB_POSTS -->|title + content| EMB
        EMB --> DB_PEMB[("PostgreSQL: post_embeddings")]
    end

    subgraph Matching["2. Ranking, Mismatch Guard & Review API (app/main.py)"]
        REQ["GET /suggestions/{post_id}/"] --> SIM["Cosine Similarity Ranking\n(app/cosine_similarity.py)"]
        DB_IEMB --> SIM
        DB_PEMB --> SIM
        SIM -->|Top 5 candidates| GUARD["Mismatch Guard (app/guard.py)\n1. Subject Match\n2. Similarity >= 0.33\n3. Confidence >= 0.70"]
        GUARD -->|Pass or Reject + Reason| DB_SUG[("PostgreSQL: suggestions")]
        DB_SUG --> REV["GET /reviews/{id}\nPOST /reviews/{id} (approve / reject)"]
        REV --> DB_REV[("PostgreSQL: review_decisions")]
    end
```

---

## Setup & Run Steps

### 1. Environment Configuration
Copy `.env.example` to `.env` and fill in your keys:
```bash
cp .env.example .env
```
- `GEMINI_API_KEY`: **Required** — Google AI Studio key (Free Tier) for tagging images via `scripts/batch_image.py`.
- `PEXELS_API_KEY`: **Required** — Pexels API key (free) for downloading the 50-image dataset into `data/images/` via `scripts/fetch_dataset.py` (`data/images/` is excluded from git via `.gitignore`).
- `DATABASE_URL`: `postgresql://postgres:password@localhost:5432/postgres` (automatically overridden to `@db:5432` inside Docker via `compose.yaml`).

### 2. Start PostgreSQL & FastAPI Server
```bash
docker compose up --build
```
The API server starts at `http://localhost:8000` (Swagger UI at `http://localhost:8000/docs`) and automatically initializes all database tables and indexes on startup.

### 3. Download Dataset & Seed Data (Images, Embeddings & Blog Posts)
With the database container running, download the 50 images from Pexels and run the idempotent batch and seeding scripts:
```bash
python -m scripts.fetch_dataset
python -m scripts.batch_image
python -m scripts.embed_images
python -m scripts.seed_posts
```
*(Or inside the container via `docker compose exec server python -m scripts.embed_images` and `docker compose exec server python -m scripts.seed_posts`.)*

---

## API Endpoints

| Method | Path | Description |
| :--- | :--- | :--- |
| `GET` | `/images` | List all tagged images with metadata and `flagged` status |
| `GET` | `/images/{image_id}` | Get a single image by ID (`404` if not found) |
| `GET` | `/posts` | List all blog posts |
| `GET` | `/posts/{post_id}` | Get a single blog post by ID (`404` if not found) |
| `POST` | `/posts` | Create a validated blog post (`409` on duplicate title, `422` on invalid input) |
| `GET` | `/suggestions/{post_id}/` | Rank images, run Mismatch Guard, persist suggestion, and return match or `"no confident match"` with reason |
| `GET` | `/reviews` | List all persisted suggestions with guard results, rejection reasons, and review decisions |
| `GET` | `/reviews/{suggestion_id}` | Inspect why a suggestion was selected or rejected |
| `POST` | `/reviews/{suggestion_id}` | Approve or reject a suggestion (`{"decision": "approved" \| "rejected"}`, `409` on duplicate review) |

---

## Evaluation

**Top-1 Precision: 13/15 (86.67%)** — measured on the 15-post hand-labeled evaluation set (`data/eval_labels.json`) via `scripts/evaluate.py`.
- All 4 out-of-corpus negative posts (`post_12`–`post_15`: sea turtle, golden eagle, African lion, PostgreSQL/Docker) are safely rejected as `None` (`no confident match`).
- The 2 strict file-level misses (`post_02` and `post_04`) still select the correct animal species (**15/15 = 100% category-level precision**, zero cross-species false positives).

Run the evaluation script:
```bash
python -m scripts.evaluate
```

---

## Limitations

**Gemini API Free Tier:** The vision pipeline (`app/vision.py` and `scripts/batch_image.py`) implements exponential-backoff retries, model fallback (`gemini-3.6-flash` / `gemini-3.5-flash-lite`), 5-second rate-limit pacing, and PostgreSQL-backed idempotency (`ON CONFLICT (file_path) DO NOTHING`). Because Gemini API Free Tier occasionally returns `503 UNAVAILABLE` / `429 RESOURCE_EXHAUSTED` under load, re-running `python -m scripts.batch_image` safely skips already-tagged images and only processes the remaining failed ones until all 50/50 images are stored. Due to the same Free Tier instability on cloud embeddings, local embeddings (`sentence-transformers` `all-MiniLM-L6-v2`) are used for deterministic `$0` semantic matching.