# Design Document

## Image Metadata Schema

```python
from pydantic import BaseModel, Field

class Metadata(BaseModel):
    category: str                                   # main entity in the image (e.g. "red_fox", "wolf", "moose", "bear", "dog")
    attributes: list[str]                           # descriptive traits or details
    caption: str                                    # readable image description
    confidence: float = Field(ge=0.0, le=1.0)       # model certainty score from 0-1
```

## Blog Post Schema (`data/blogs/posts.json`)

```python
from pydantic import BaseModel

class BlogPostSeed(BaseModel):
    id: str                                         # stable post identifier (e.g. "post_01")
    title: str                                      # blog post headline
    content: str                                    # full post body used for embedding
    expected_category: str | None = None            # expected category ("red_fox", "wolf", etc., or None if no match)
    test_purpose: str                               # eval-only scenario label (e.g. probe_2_and_3, probe_4)
```

## Matching Strategy

For each post, compute the cosine similarity between the post's embedding (generated from `title + content`) and every image's caption embedding. Cosine similarity measures how closely two embeddings point in the same direction, capturing semantic closeness regardless of exact wording.

**Examples from `posts.json`:**
* *Red Fox* matches *Vulpes vulpes* (`post_02`) and *orange-coated forest fox* (`post_03`).
* *Wolf* matches *Canis lupus* (`post_05`), *Bear* matches *Ursus arctos* (`post_09`), and *Moose* matches *Alces alces* (`post_11`).

```text
cosine_similarity = (A · B) / (|A| × |B|)
```

Rank images by similarity score and evaluate the top 5 candidates per post through the Mismatch Guard.

## Mismatch Guard

Before a ranked image is suggested, it must pass **three** checks in order. The first check it fails determines the rejection reason. Checks after that point are not evaluated.

| # | Check                | Rejects When                                                           | Reason Shown                                                                         |
| - | -------------------- | ---------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| 1 | Subject Match        | `post.expected_category is None` or `image.category != post.expected_category` | `"Subject mismatch: expected <post.expected_category>, detected <image.category>"` |
| 2 | Similarity Threshold | `similarity < sim_threshold`                                           | `"Similarity score below threshold: image content is not relevant enough"`           |
| 3 | Confidence Threshold | `confidence < con_threshold`                                           | `"Vision model confidence too low to accept this image safely"`                      |

* An image that passes all three checks is accepted and returned as the suggestion.
* If no candidate passes (for example `post_12`–`post_15` where `expected_category` is `None`), the post receives `no confident match` along with the rejection reason from the top-ranked candidate's failed check.

## Database Design

```sql
CREATE TABLE images (
  id SERIAL PRIMARY KEY,
  category TEXT NOT NULL,
  attributes TEXT[] NOT NULL,
  caption TEXT NOT NULL,
  confidence FLOAT NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
  file_path TEXT NOT NULL UNIQUE,
  created_at TIMESTAMP DEFAULT now()
);

CREATE TABLE posts (
  id SERIAL PRIMARY KEY,
  title TEXT NOT NULL UNIQUE,
  content TEXT NOT NULL,
  expected_category TEXT,                           -- nullable for out-of-corpus / no-match posts
  created_at TIMESTAMP DEFAULT now()
);

CREATE TABLE image_embeddings (
  id SERIAL PRIMARY KEY,
  image_id INT NOT NULL REFERENCES images(id) ON DELETE CASCADE,
  vector FLOAT[] NOT NULL,
  created_at TIMESTAMP DEFAULT now()
);

CREATE TABLE post_embeddings (
  id SERIAL PRIMARY KEY,
  post_id INT NOT NULL REFERENCES posts(id) ON DELETE CASCADE,
  vector FLOAT[] NOT NULL,
  created_at TIMESTAMP DEFAULT now()
);

CREATE TABLE suggestions (
  id SERIAL PRIMARY KEY,
  image_id INT NOT NULL REFERENCES images(id) ON DELETE CASCADE,
  post_id INT NOT NULL REFERENCES posts(id) ON DELETE CASCADE,
  similarity_score FLOAT NOT NULL,
  guard_passed BOOLEAN NOT NULL,
  rejection_reason TEXT,
  created_at TIMESTAMP DEFAULT now(),
  UNIQUE (post_id, image_id)
);

CREATE TABLE review_decisions (
  id SERIAL PRIMARY KEY,
  suggestion_id INT NOT NULL REFERENCES suggestions(id) ON DELETE CASCADE UNIQUE,
  decision TEXT NOT NULL CHECK (decision IN ('approved', 'rejected')),
  reviewed_at TIMESTAMP DEFAULT now()
);

CREATE INDEX idx_suggestions_post_id ON suggestions(post_id);
CREATE INDEX idx_suggestions_image_id ON suggestions(image_id);
CREATE INDEX idx_image_embeddings_image_id ON image_embeddings(image_id);
CREATE INDEX idx_post_embeddings_post_id ON post_embeddings(post_id);
CREATE INDEX idx_review_decisions_suggestion_id ON review_decisions(suggestion_id);
```

## Dataset

### Images (`data/images/`)
50 images across 5 animal categories, gathered from Pexels via `scripts/fetch_dataset.py` to keep the project reproducible.

* **Categories:** `moose`, `bear`, `red_fox`, `wolf`, `dog` (10 images each)
* **Source:** Pexels API (free tier, licensed for reuse); see [Pexels License](https://www.pexels.com/license/)
* **Reproduction:** Run `scripts/fetch_dataset.py` with a valid `PEXELS_API_KEY` in `.env`. Images are saved to `data/images/<category>/`.

### Blog Posts & Eval Set (`data/blogs/posts.json`)
15 structured blog posts designed to cover seeding, semantic retrieval, mismatch rejection, and top-1 precision evaluation:

| Group | Posts | Purpose |
| ----- | ----- | ------- |
| **Red Fox (3)** | `post_01`, `post_02`, `post_03` | Core fox match (Probe 2), wolf-on-fox rejection (Probe 3), scientific name (*Vulpes vulpes*) & attribute matching |
| **Wolf (2)** | `post_04`, `post_05` | Standard wolf match + scientific name (*Canis lupus*) semantic match |
| **Dog (2)** | `post_06`, `post_07` | Domestic dog vs. wild canid disambiguation + companion/training semantic match |
| **Bear (2)** | `post_08`, `post_09` | Standard bear match + scientific name (*Ursus arctos*) semantic match |
| **Moose (2)** | `post_10`, `post_11` | Standard moose match + scientific name (*Alces alces*) semantic match |
| **No-Match Negatives (4)** | `post_12`–`post_15` | Probe 4 (`no confident match`): sea turtle (marine), golden eagle (bird), African lion (hard-negative predator), PostgreSQL/Docker (off-domain) |
