# AI Image Understanding & Content Matching Engine

An AI backend service that inspects a 50-image wildlife library with **Gemini Vision**, validates structured metadata with **Pydantic**, embeds image captions and blog posts into a shared semantic vector space (`sentence-transformers` `all-MiniLM-L6-v2`), ranks candidates by **cosine similarity**, and filters every recommendation through a three-check **Mismatch Guard** before surfacing it to a human review API.

See [DESIGN.md](DESIGN.md) for the full design document, database schema, guard rules, and dataset breakdown.

---

## Architecture

Two parallel semantic streams (vision-tagged image captions and blog posts) meet in a shared 384-dimensional vector space, where every ranked recommendation must clear the three-check **Mismatch Guard** before persistence and human review.

```mermaid
flowchart TB
    subgraph Ingestion["1. Data Ingestion & Local Embeddings (Batch CLI)"]
        PEX["Pexels API"] --> FETCH["scripts/fetch_dataset.py<br/>50 wildlife photos → data/images/"]
        FETCH --> BATCH["scripts/batch_image.py<br/>Retries · Pacing · Idempotent"]
        BATCH --> VIS["Gemini Vision (app/vision.py)<br/>Pydantic Metadata + Cost Log"]
        EMB_IMG["scripts/embed_images.py<br/>all-MiniLM-L6-v2 (384-d)"]
        POST_JSON["data/blogs/posts.json<br/>15 blog posts"] --> SEED["scripts/seed_posts.py<br/>all-MiniLM-L6-v2 (384-d)"]
    end

    subgraph Storage["2. Persistence Layer (PostgreSQL)"]
        DB_IMG[("images<br/>tags · caption · flagged")]
        DB_IEMB[("image_embeddings<br/>384-d vectors")]
        DB_POST[("posts")]
        DB_PEMB[("post_embeddings<br/>384-d vectors")]
        DB_SUG[("suggestions<br/>score · guard_passed · reason")]
        DB_REV[("review_decisions<br/>approved / rejected")]
    end

    subgraph Serving["3. Matching Engine, Mismatch Guard & Review API (FastAPI)"]
        CLIENT(["API Client / Swagger UI"]) --> REQ["GET /suggestions/{post_id}/"]
        REQ --> SIM["Cosine Similarity Ranking<br/>app/cosine_similarity.py (Top 5)"]
        SIM --> GUARD{"Mismatch Guard (app/guard.py)<br/>1. Subject Match<br/>2. Similarity >= 0.33<br/>3. Confidence >= 0.70"}
        GUARD -->|Pass| MATCH["Match Response<br/>suggested image + score"]
        GUARD -->|All Fail| REJECT["No Confident Match<br/>top candidate + rejection reason"]
        CLIENT --> REV_API["GET /reviews/{id}<br/>POST /reviews/{id}"]
    end

    VIS --> DB_IMG
    DB_IMG --> EMB_IMG
    EMB_IMG --> DB_IEMB
    SEED --> DB_POST
    SEED --> DB_PEMB
    DB_IEMB --> SIM
    DB_PEMB --> SIM
    MATCH --> DB_SUG
    REJECT --> DB_SUG
    DB_SUG --> REV_API
    REV_API --> DB_REV

    classDef external fill:#1e293b,stroke:#64748b,color:#f8fafc
    classDef ai fill:#312e81,stroke:#818cf8,color:#eef2ff
    classDef db fill:#0f172a,stroke:#38bdf8,color:#e0f2fe
    classDef api fill:#134e4a,stroke:#2dd4bf,color:#f0fdfa
    classDef guard fill:#78350f,stroke:#fbbf24,color:#fef3c7
    classDef pass fill:#14532d,stroke:#4ade80,color:#dcfce7
    classDef fail fill:#7f1d1d,stroke:#f87171,color:#fee2e2

    class PEX,POST_JSON,CLIENT external
    class FETCH,BATCH,VIS,EMB_IMG,SEED ai
    class DB_IMG,DB_IEMB,DB_POST,DB_PEMB,DB_SUG,DB_REV db
    class REQ,SIM,REV_API api
    class GUARD guard
    class MATCH pass
    class REJECT fail
```

<details>
<summary>ASCII Architecture Diagram (plain-text view)</summary>

```text
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ 1. DATA INGESTION & LOCAL EMBEDDINGS (Batch CLI)                                    │
│                                                                                     │
│   Pexels API ─▶ scripts/fetch_dataset.py ─▶ data/images/ (50 photos)                │
│                                                  │                                  │
│                                     scripts/batch_image.py                          │
│                                     (Gemini Vision + Pydantic + Cost Log)           │
│                                                  ▼                                  │
│                                     PostgreSQL: images                              │
│                                                  │                                  │
│                                     scripts/embed_images.py (all-MiniLM-L6-v2)      │
│                                                  ▼                                  │
│                                     PostgreSQL: image_embeddings (384-d)            │
│                                                                                     │
│   data/blogs/posts.json (15) ─▶ scripts/seed_posts.py ─▶ PostgreSQL: posts          │
│                                                       └▶ post_embeddings            │
└──────────────────────────────────────────────────┬──────────────────────────────────┘
                                                   │
                                                   ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ 2. MATCHING ENGINE, MISMATCH GUARD & REVIEW API (FastAPI)                           │
│                                                                                     │
│   API Client ─▶ GET /suggestions/{post_id}/                                         │
│                      │                                                              │
│                      ▼                                                              │
│                 Cosine Similarity Ranking (post_vector × image_vectors → Top 5)     │
│                      │                                                              │
│                      ▼                                                              │
│                 Mismatch Guard (app/guard.py)                                       │
│                   [1] Subject Match    (image.category == post.expected_category)   │
│                   [2] Similarity Gate  (similarity >= 0.33)                         │
│                   [3] Confidence Gate  (confidence >= 0.70)                         │
│                      │                                                              │
│                      ├── Pass ─────▶ {"result": "match", "suggestion": ...}         │
│                      └── All Fail ─▶ {"result": "no confident match", "reason": ...}│
│                      │                                                              │
│                      ▼                                                              │
│                 PostgreSQL: suggestions ──▶ GET /reviews/{id}                       │
│                                         └── POST /reviews/{id} ─▶ review_decisions  │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

</details>

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