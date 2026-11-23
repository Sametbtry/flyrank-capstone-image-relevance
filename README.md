# Image Relevance

AI Image Understanding & Content Matching Engine. Tags images with a vision model, matches them to blog posts by meaning, and rejects bad matches with an explanation.

## Limitations

**Gemini API Free Tier:** The vision pipeline (`app/vision.py` and `app/batch.py`) implements exponential-backoff retries, model fallback, rate-limit pacing, and PostgreSQL-backed idempotency (`ON CONFLICT (file_path) DO NOTHING`). Because Gemini API Free Tier occasionally returns `503 UNAVAILABLE` / `429 RESOURCE_EXHAUSTED` under load, re-running `python -m app.batch` safely skips already-tagged images and only processes the remaining failed ones.