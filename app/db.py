import os
from dotenv import load_dotenv
import psycopg
from psycopg_pool import AsyncConnectionPool

load_dotenv()

DB_URL = os.getenv("DATABASE_URL")
if not DB_URL:
    raise RuntimeError("DATABASE_URL not set in .env")

INIT_DB ="""
CREATE TABLE IF NOT EXISTS images (
  id SERIAL PRIMARY KEY,
  category TEXT NOT NULL,
  attributes TEXT[] NOT NULL,
  caption TEXT NOT NULL,
  confidence FLOAT NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
  flagged BOOLEAN NOT NULL DEFAULT FALSE,
  file_path TEXT NOT NULL UNIQUE,
  created_at TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS posts (
  id SERIAL PRIMARY KEY,
  title TEXT NOT NULL UNIQUE,
  content TEXT NOT NULL,
  expected_category TEXT,
  created_at TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS image_embeddings (
  id SERIAL PRIMARY KEY,
  image_id INT NOT NULL REFERENCES images(id) ON DELETE CASCADE,
  vector FLOAT[] NOT NULL,
  created_at TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS post_embeddings (
  id SERIAL PRIMARY KEY,
  post_id INT NOT NULL REFERENCES posts(id) ON DELETE CASCADE,
  vector FLOAT[] NOT NULL,
  created_at TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS suggestions (
  id SERIAL PRIMARY KEY,
  image_id INT NOT NULL REFERENCES images(id) ON DELETE CASCADE,
  post_id INT NOT NULL REFERENCES posts(id) ON DELETE CASCADE,
  similarity_score FLOAT NOT NULL,
  guard_passed BOOLEAN NOT NULL,
  rejection_reason TEXT,
  created_at TIMESTAMP DEFAULT now(),
  UNIQUE (post_id, image_id)
);

CREATE TABLE IF NOT EXISTS review_decisions (
  id SERIAL PRIMARY KEY,
  suggestion_id INT NOT NULL REFERENCES suggestions(id) ON DELETE CASCADE UNIQUE,
  decision TEXT NOT NULL CHECK (decision IN ('approved', 'rejected')),
  reviewed_at TIMESTAMP DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_suggestions_post_id ON suggestions(post_id);
CREATE INDEX IF NOT EXISTS idx_suggestions_image_id ON suggestions(image_id);
CREATE INDEX IF NOT EXISTS idx_image_embeddings_image_id ON image_embeddings(image_id);
CREATE INDEX IF NOT EXISTS idx_post_embeddings_post_id ON post_embeddings(post_id);
CREATE INDEX IF NOT EXISTS idx_review_decisions_suggestion_id ON review_decisions(suggestion_id);

"""
pool = AsyncConnectionPool(conninfo=DB_URL, open=False)


async def get_connection():
    """Connect to the database and yield a connection from the pool"""
    """Connect to the database and yield a connection from the pool."""
    async with pool.connection() as conn:
        yield conn


def get_sync_connection() -> psycopg.Connection:
    """Open a synchronous PostgreSQL connection for batch/CLI scripts."""
    return psycopg.connect(DB_URL)


def init_db() -> None:
    """Apply the initial schema migration idempotently."""
    sql = INIT_DB
    with get_sync_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()
