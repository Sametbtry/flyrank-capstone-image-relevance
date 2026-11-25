# Build Log — AI-Assisted Development

This log documents where AI tools helped, where suggestions were wrong or over-engineered, and what I changed while building this capstone.

---

## Phase 1: Design, Dataset & Docker Setup

- **Where AI helped:** Structuring the 15-post evaluation corpus (`data/blogs/posts.json`), and setting up FastAPI + PostgreSQL with Docker Compose.
- **Where it was wrong / issues hit:**
  - Running `docker compose up` failed with `connection to server at "127.0.0.1", port 5432 failed: Connection refused` because `.env` used `localhost:5432`, which resolves to the `server` container itself rather than the `db` service.
  - `.env.example` was missing the `DATABASE_URL=` prefix on line 3.
- **What I changed:**
  - Kept `localhost:5432` in `.env` for local CLI scripts and overrode `DATABASE_URL` inside `compose.yaml` (`@db:5432`) for the container network.
  - Fixed `.env.example` with proper placeholder keys.

---

## Phase 2: Vision Pipeline, Batch Processing & Cost Tracking

- **Where AI helped:** Building the Gemini structured-output call with Pydantic `Metadata` validation (`app/vision.py`), designing PostgreSQL-backed idempotent batch processing, and adding token-based cost estimation.
- **Where it was wrong / what I rejected:**
  - **Gemini Free Tier 503/429 errors:** Initial single-pass batch runs hit transient `503`/`429` errors (36/50 succeeded on the first run).
  - **Manual metadata fallback rejected:** Instead of replacing live vision tagging with a pre-generated JSON file, I kept real Gemini Vision outputs.
  - **Over-engineered cost tracking rejected:** AI initially created a separate `app/costs.py` module writing to `logs/costs.jsonl` with a budget guard file reader. Since this runs on the Free Tier, that was unnecessary complexity.
- **What I changed:**
  - Inlined the PostgreSQL schema (`INIT_DB`) directly in `app/db.py` instead of using a separate `migrations/001_init.sql` file.
  - Moved the batch tagger to `scripts/batch_image.py`, keeping `MAX_RETRIES = 3`, `5s` rate-limit pacing, `2 ** attempt` exponential backoff, and model fallback (`gemini-3.6-flash` / `gemini-3.5-flash-lite`).
  - Made the batch job idempotent against PostgreSQL (`ON CONFLICT (file_path) DO NOTHING`), immediately committing each tagged image and skipping already-stored images on re-runs until all 50/50 images were tagged cleanly.
  - Removed `app/costs.py` and kept a minimal `_log_cost()` helper using Python's standard `logging` module directly in `app/vision.py`.

---

## Phase 3: Local Embeddings, Ranking & Mismatch Guard

- **Where AI helped:** Implementing idempotent embedding scripts (`scripts/embed_images.py`, `scripts/seed_posts.py`), the cosine similarity ranking endpoint (`app/main.py`), and the three-check mismatch guard (`app/guard.py`).
- **Where it was wrong / issues hit:**
  - Gemini Embedding API suffered from the same Free Tier instability as the vision API.
  - `app/cosine_similarity.py` initially returned `np.float64` instead of a native Python `float`.
  - `app/main.py` initially defined `app = FastAPI()` twice (and later referenced `lifespan` before defining it), which dropped the `GET /` route / raised a `NameError`.
  - Inside Docker, the non-root `appuser` had `--home /nonexistent`, which would cause `sentence-transformers` to fail with `PermissionError` when caching models.
- **What I changed:**
  - Switched exclusively to local embeddings (`sentence-transformers` with `all-MiniLM-L6-v2`) in `app/embeddings.py` and documented this in `README.md`.
  - Converted `cosine_similarity` return value to `float(result)`.
  - Fixed `lifespan` ordering in `app/main.py` and added `ENV HF_HOME=/app/.cache/huggingface` with proper ownership in `Dockerfile`.
  - Kept `app/guard.py` minimal (~30 lines) implementing the exact three checks from `DESIGN.md` in order: subject match → similarity threshold → confidence threshold.

---

## Phase 4: Validated API, Suggestion Persistence & Review Workflow

- **Where AI helped:** Adding Pydantic request/response schemas (`app/schema.py`), persisting suggestion outcomes in `GET /suggestions/{post_id}/`, and implementing the review endpoints (`GET /reviews`, `GET /reviews/{suggestion_id}`, `POST /reviews/{suggestion_id}`).
- **Where it was wrong / issues hit:**
  - **Unnecessary endpoint:** AI initially added `POST /images` (`ImageCreate`), which was redundant since images are ingested via `scripts/batch_image.py`.
  - **Skipped `SERIAL` IDs in `suggestions`:** Initial `_save_suggestion` used `INSERT ... ON CONFLICT (post_id, image_id) DO UPDATE`. In PostgreSQL, `ON CONFLICT` consumes `nextval('suggestions_id_seq')` before checking the conflict, causing suggestion IDs to jump whenever the same post was queried repeatedly.
- **What I changed:**
  - Removed `POST /images` and `ImageCreate`, kept `POST /reviews/{suggestion_id}` (required for the approve/reject workflow), and added `GET /images/{image_id}` and `GET /posts/{post_id}` for single-record lookup.
  - Replaced `INSERT ... ON CONFLICT DO UPDATE` in `_save_suggestion` with a `SELECT` check followed by `UPDATE` (when the pair exists) or `INSERT` (when new), keeping `suggestions.id` sequential.
  - Keyed `data/eval_labels.json` by post `title` and `file_path` (instead of auto-increment database IDs) and evaluated directly over embeddings + `check_mismatch_guard` in `scripts/evaluate.py` (`SIM_THRESHOLD = 0.33`) so evaluation runs deterministically on any database without requiring prior API calls.
  - Added `capstone.yaml`, `LICENSE`, and the architecture diagram + run/seed documentation in `README.md` to complete the Section 11 submission pack.
