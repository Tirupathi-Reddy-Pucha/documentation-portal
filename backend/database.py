import datetime
import os
import uuid

# python-dotenv is used to load configuration variables from a .env file
from dotenv import load_dotenv

# Neo4j is our Graph Database. We use it to store concepts and their relationships.
from neo4j import GraphDatabase

# pgvector is a PostgreSQL extension that allows us to store and search
# mathematical vectors (embeddings) directly in the database.
from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, DateTime, String, Text
from sqlalchemy.dialects.postgresql import UUID

# SQLAlchemy is our Object-Relational Mapper (ORM). It lets us interact with
# PostgreSQL using Python classes instead of raw SQL queries.
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# -----------------------------------------------------------------------------
# Configuration Loading
# -----------------------------------------------------------------------------
# Load environment variables from the .env file located at the project root.
# os.path.dirname(__file__) gives us the 'backend/' folder.
# '..' moves us one level up to the root folder where '.env' is located.
load_dotenv(
    dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"), override=True
)


# 1. PostgreSQL Setup — Stores document metadata and vector embeddings

# Fetch database credentials from environment variables, providing safe defaults
POSTGRES_USER = os.getenv("POSTGRES_USER", "cluser")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "clpassword")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5434")
POSTGRES_DB = os.getenv("POSTGRES_DB", "curriculumlens")

# Construct the connection string required by SQLAlchemy
DATABASE_URL = (
    f"postgresql+asyncpg://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
    f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)

# The 'engine' is the core interface to the database
engine = create_async_engine(DATABASE_URL, echo=False)

# A 'session' is an ongoing transaction with the database. We use an async session.
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# 'Base' is the parent class for all our database models
Base = declarative_base()


# -----------------------------------------------------------------------------
# Database Models (Tables)
# -----------------------------------------------------------------------------


class Document(Base):
    """
    Represents an uploaded PDF document.
    Table name: documents
    """

    __tablename__ = "documents"

    # Primary key: A unique identifier for every document
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # The original name of the uploaded file
    filename = Column(String, nullable=False)
    # The type of document ("syllabus" or "pyq")
    doc_type = Column(String, nullable=False, default="pyq")
    # The department (for syllabus docs, e.g., "CSE")
    department = Column(String, nullable=True)
    # The year of the syllabus (for syllabus docs, e.g., "2023")
    year = Column(String, nullable=True)
    # The academic course code associated with this document (e.g., CSE101)
    course_code = Column(String, nullable=True)
    # Timestamp of when the document was uploaded
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class DocumentChunk(Base):
    """
    Represents a specific paragraph or section of text extracted from a document.
    Table name: document_chunks
    """

    __tablename__ = "document_chunks"

    # Primary key: A unique identifier for this specific chunk
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # Foreign key-like relationship to the parent Document
    document_id = Column(UUID(as_uuid=True), nullable=False)
    # The actual text content of this chunk
    content = Column(Text, nullable=False)
    # The course code, duplicated here for easier querying
    course_code = Column(String, nullable=True)

    # Marks whether this chunk was extracted as plain text or described from a visual
    # element (equation, diagram, circuit) by the vision model. Values: "text" or "visual".
    content_type = Column(String, default="text")

    # The vector embedding representing the semantic meaning of the text content.
    # The dimension is 768 to perfectly match the output of the 'nomic-embed-text' model.
    embedding = Column(Vector(768), nullable=True)


class ProcessingReport(Base):
    """
    Stores per-document PYQ extraction and mapping audit report as a JSON payload.
    Table name: processing_reports
    """
    __tablename__ = "processing_reports"

    id          = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    course_code = Column(String, nullable=True)
    report_json = Column(Text, nullable=False, default="{}")
    created_at  = Column(DateTime, default=datetime.datetime.utcnow)

    # Fast Approximate Nearest Neighbor (ANN) index for vector similarity search
    # We use vector_cosine_ops for cosine similarity matching
    __table_args__ = (
        Index(
            'ix_document_chunks_embedding_hnsw',
            embedding,
            postgresql_using='hnsw',
            postgresql_with={'m': 16, 'ef_construction': 64},
            postgresql_ops={'embedding': 'vector_cosine_ops'}
        ),
    )


class ProcessingReport(Base):
    """
    Stores the per-document extraction and mapping report produced after each
    PYQ or syllabus upload. Allows the admin dashboard to display exact counts,
    skip reasons, per-stage timings, and partial-completion details without
    re-querying Neo4j.

    Table name: processing_reports
    """
    __tablename__ = "processing_reports"

    id          = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    course_code = Column(String, nullable=True)
    # JSON payload matching the PYQProcessingReport or SyllabusProcessingReport dataclass.
    # Stored as TEXT rather than native JSONB so we don't need an extra extension.
    report_json = Column(Text, nullable=False, default="{}")
    created_at  = Column(DateTime, default=datetime.datetime.utcnow)


# -----------------------------------------------------------------------------
# Database Dependency Injection
# -----------------------------------------------------------------------------
async def get_db():
    """
    A helper function that provides a database session to our FastAPI routes.
    It ensures the session is properly closed after the request finishes.
    """
    async with AsyncSessionLocal() as session:
        yield session


# 2. Neo4j Setup — Stores the Curriculum Knowledge Graph

# Fetch Neo4j connection details from environment variables
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "clpassword")

# Create a single, shared driver instance to communicate with the Neo4j database
neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def get_neo4j():
    """
    Returns the shared Neo4j driver instance.
    """
    return neo4j_driver
