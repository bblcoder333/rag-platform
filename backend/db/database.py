import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:password@localhost:5432/ragplatform"
)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)


def init_db():
    """Create tables and enable pgvector extension."""
    with engine.connect() as conn:
        # Enable pgvector
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

        # Create documents table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS documents (
                id SERIAL PRIMARY KEY,
                source TEXT NOT NULL,
                file_path TEXT,
                num_pages INTEGER,
                file_size_kb FLOAT,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """))

        # Create chunks table with vector column
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS chunks (
                id SERIAL PRIMARY KEY,
                document_id INTEGER REFERENCES documents(id),
                chunk_index INTEGER,
                text TEXT NOT NULL,
                strategy TEXT,
                chunk_size INTEGER,
                char_count INTEGER,
                embedding vector(384),
                created_at TIMESTAMP DEFAULT NOW()
            )
        """))

        # Create index for fast similarity search
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS chunks_embedding_idx
            ON chunks
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 10)
        """))

        conn.commit()
        print("Database initialized successfully.")
        print("Tables created: documents, chunks")
        print("Vector index created on chunks.embedding")


if __name__ == "__main__":
    init_db()