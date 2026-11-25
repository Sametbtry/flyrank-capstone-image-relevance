# Image Relevance

AI Image Understanding & Content Matching Engine. Tags images with a vision model, matches them to blog posts by meaning, and rejects bad matches with an explanation.

## Evaluation

**Top-1 Precision: 13/15 (86.67%)** — measured on the 15-post hand-labeled evaluation set (`data/eval_labels.json`) via `scripts/evaluate.py`.
- All 4 out-of-corpus negative posts (`post_12`–`post_15`) are safely rejected as `None` (`no confident match`), and the 2 misses (`post_02` and `post_04`) still select the correct animal species (**15/15 = 100% category-level precision**, zero cross-species mismatches).

Run the evaluation script:
```bash
python -m scripts.evaluate
```

## Limitations

**Gemini API Free Tier:** The vision pipeline (`app/vision.py` and `scripts/batch_image.py`) implements exponential-backoff retries, model fallback, rate-limit pacing, and PostgreSQL-backed idempotency (`ON CONFLICT (file_path) DO NOTHING`). Because Gemini API Free Tier occasionally returns `503 UNAVAILABLE` / `429 RESOURCE_EXHAUSTED` under load, re-running `python -m scripts.batch_image` safely skips already-tagged images and only processes the remaining failed ones. The same Free Tier instability applies to Gemini embeddings, which is why local embeddings (`sentence-transformers` `all-MiniLM-L6-v2`) are used.