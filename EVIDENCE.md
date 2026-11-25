# Evidence Checklist

Each requirement from Section 6 of the Capstone Brief is tracked below with real command output or verification proof. Checked boxes (`[x]`) have verified proof; unchecked boxes (`[ ]`) will be filled with live DB/API output or completed in Phase 4.

---

## AI Processing

- [x] **Vision model produces structured output validated against a schema; invalid responses are never trusted.**
  - Implementation: `app/schema.py` (`Metadata` Pydantic model) + `app/vision.py` (`Metadata.model_validate(response.parsed)`).
  - Proof (valid input accepted, invalid `confidence=1.5` rejected):
  ```text
  VALID_SCHEMA: {'category': 'red_fox', 'attributes': ['orange fur', 'forest'], 'caption': 'A red fox in a forest', 'confidence': 0.94}
  INVALID_SCHEMA_REJECTED: Input should be less than or equal to 1
  ```

- [x] **Low-confidence classifications are flagged instead of accepted.**
  - Implementation: `app/db.py` (`images.flagged BOOLEAN`) + `scripts/batch_image.py` (`flagged = metadata.confidence < 0.7`) + `app/guard.py` (`CONF_THRESHOLD = 0.70`).
  - Guard proof:
  ```text
  GUARD_LOW_CONF: (False, 'Vision model confidence too low to accept this image safely')
  ```

- [x] **Images are processed through a batch background job with retries.**
  - Implementation: `scripts/batch_image.py` (`tag_with_retries`, `MAX_RETRIES = 3`, `FALLBACK_MODELS`, `5s` pacing, `2 ** attempt` exponential backoff, PostgreSQL idempotency via `ON CONFLICT (file_path) DO NOTHING`).
    ```text
    [48/50] [SKIP] data/images/wolf/wolf_08.jpg (already in DB)
    [49/50] [SKIP] data/images/wolf/wolf_09.jpg (already in DB)
    [50/50] [SKIP] data/images/wolf/wolf_10.jpg (already in DB)

    Batch complete — Tagged: 0 | Skipped: 50 | Failed: 0 | Total in DB: 50
    ```

- [x] **Vision and embedding costs are tracked per call.**
  - Implementation: `app/vision.py` (`_log_cost` logs prompt/candidate tokens and estimated USD per call via Python `logging`); embeddings run locally via `sentence-transformers` (`all-MiniLM-L6-v2`, `$0`).
  ```text
      INFO:httpx:HTTP Request: POST https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:generateContent "HTTP/1.1 200 OK"
      INFO:vision_costs:data\images\bear\bear_01.jpg: 1140 in / 49 out tokens | estimated: $0.000464
      category='bear' attributes=['brown fur', 'wet fur', 'in water', 'wildlife', 'close-up'] caption='A brown bear partially submerged in water looks directly at another bear in the foreground.' confidence=0.96
  ```

## Matching System

- [x] **Image and post embeddings are stored; posts return ranked image suggestions.**
  - Implementation: `scripts/embed_images.py`, `scripts/seed_posts.py`, `app/main.py` (`GET /suggestions/{post_id}/`).
    ```text
    {
      "post_id": 1,
      "result": "match",
      "suggestion": {
        "image_id": 46,
        "file_path": "data/images/red_fox/red_fox_10.jpg",
        "caption": "Two red foxes resting on green grass outdoors",
        "category": "red_fox",
        "confidence": 0.98,
        "similarity": 0.7020194470176513
      }
    }
    ```

- [x] **Semantic matching works for equivalent concepts — "red fox" matches "Vulpes vulpes".**
  - Implementation: `app/embeddings.py` (`all-MiniLM-L6-v2`) + `app/cosine_similarity.py`.
  - Proof (`"A red fox standing in a temperate forest"` vs `"Understanding Vulpes vulpes: wild fox species"` vs `"PostgreSQL and Docker setup"`):
  ```text
  SIM_FOX_VULPES: 0.5395
  SIM_FOX_DB: -0.01
  ```

---

## Safety Layer

- [x] **The mismatch guard rejects incorrect recommendations — the wolf-on-a-fox-post scenario provably fails.**
  - Implementation: `app/guard.py` (`check_mismatch_guard`).
  - Proof:
  ```text
  GUARD_MATCH: (True, None)
  GUARD_WOLF_ON_FOX: (False, 'Subject mismatch: expected red_fox, detected wolf')
  ```

- [x] **Rejections include a human-readable explanation.**
  - Proof (all three guard checks return clear explanations):
  ```text
  GUARD_WOLF_ON_FOX: (False, 'Subject mismatch: expected red_fox, detected wolf')
  GUARD_LOW_SIM: (False, 'Similarity score below threshold: image content is not relevant enough')
  GUARD_LOW_CONF: (False, 'Vision model confidence too low to accept this image safely')
  ```

- [x] **When no image clears the bar, the system answers "no confident match" with reasons.**
  - Implementation: `app/main.py` returns `{"post_id": ..., "result": "no confident match", "reason": top_reason, "top_candidate": top}` when none of the top 5 candidates pass `check_mismatch_guard`.
  - Proof (`expected_category=None`, e.g. `post_12`–`post_15`):
  ```text
    {
      "post_id": 12,
      "result": "no confident match",
      "reason": "Subject mismatch: expected None, detected red_fox",
      "top_candidate": {
        "image_id": 46,
        "file_path": "data/images/red_fox/red_fox_10.jpg",
        "caption": "Two red foxes resting on green grass outdoors",
        "category": "red_fox",
        "confidence": 0.98,
        "similarity": 0.19590046859252305
      }
    }
  ```

---

## Backend

- [x] **Database models for images, tags, embeddings, posts, suggestions, approvals/rejections — with the required indexes.**
  - Proof: `app/db.py` (`INIT_DB`) defines 6 tables (`images`, `posts`, `image_embeddings`, `post_embeddings`, `suggestions`, `review_decisions`) and 5 indexes (`idx_suggestions_post_id`, `idx_suggestions_image_id`, `idx_image_embeddings_image_id`, `idx_post_embeddings_post_id`, `idx_review_decisions_suggestion_id`).

- [x] **API endpoints validated; the review workflow (approve / reject / inspect why) exists.**
  - Implementation: `app/schema.py` (`ImageOut`, `PostCreate`, `PostOut`, `ReviewCreate`, `ReviewOut`) + `app/main.py` (`GET /images`, `GET /images/{image_id}`, `GET /posts`, `GET /posts/{post_id}`, `POST /posts`, `GET /suggestions/{post_id}/`, `GET /reviews`, `GET /reviews/{suggestion_id}`, `POST /reviews/{suggestion_id}`).
  - Proof (registered endpoints & Pydantic boundary validation rejecting invalid input cleanly):
  ```text
  ROUTES: GET /images, GET /images/{image_id}, GET /posts, GET /posts/{post_id}, GET /reviews, GET /reviews/{suggestion_id}, GET /suggestions/{post_id}/, POST /posts, POST /reviews/{suggestion_id}
  INVALID_REVIEW_422: Input should be 'approved' or 'rejected'
  INVALID_POST_422: String should have at least 1 character
  ```

---

## Quality & Documentation

- [ ] **A small labeled evaluation dataset measures top-1 precision — the number is in your README.**
  - *Phase 4 — pending (`Stage 4.4`).*

- [ ] **README with architecture explanation and diagram; the required files from Section 11 present.**
  - *Phase 4 — pending (`Stage 5.0`: architecture diagram in `README.md` + `capstone.yaml`).*
