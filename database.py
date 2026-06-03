import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import PGVector

# Load environment variables from the .env file BEFORE doing anything else
load_dotenv()

# --- Configuration & Shared State ---
DB_CONNECTION_STRING = os.getenv("DATABASE_URL")
if not DB_CONNECTION_STRING:
    raise ValueError("CRITICAL ERROR: DATABASE_URL is not set in the .env file.")

COLLECTION_NAME = "lesson_plans"

# Global variables for lazy loading
_embeddings = None
_vector_store = None

_embeddings = None
_vector_stores = {}

def get_vector_store(collection_name: str = COLLECTION_NAME):
    """
    Dependency to lazy-load the embedding model and vector store for a specific collection.
    Imported by our routers so they share the same memory instance.
    """
    global _embeddings, _vector_stores
    if collection_name not in _vector_stores:
        print(f"Lazy Loading Embedding Model for collection '{collection_name}'... (This will take a moment on the first request)")
        if _embeddings is None:
            # UPDATED: Swapped to a much lighter multilingual model (~470MB)
            # This handles Hindi and Sanskrit perfectly without hogging your RAM or bandwidth.
            _embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")
        
        _vector_stores[collection_name] = PGVector(
            connection_string=DB_CONNECTION_STRING,
            embedding_function=_embeddings,
            collection_name=collection_name,
            use_jsonb=True, 
        )
        print(f"System Ready! Vector Store for '{collection_name}' Initialized.")
    return _vector_stores[collection_name]


def get_collection_for_input(class_name: str, subject: str, chapter_name: str) -> str:
    """
    Dynamically queries the database to find which collection contains the requested
    class, subject, and chapter. Falls back to default 'lesson_plans' if not found.
    """
    import psycopg2
    try:
        conn = psycopg2.connect(DB_CONNECTION_STRING)
        cur = conn.cursor()
        
        # 1. Try exact match (class, subject, chapter)
        cur.execute("""
            SELECT c.name
            FROM langchain_pg_collection c
            JOIN langchain_pg_embedding e ON c.uuid = e.collection_id
            WHERE e.cmetadata->>'class_name' = %s
              AND e.cmetadata->>'subject' = %s
              AND e.cmetadata->>'chapter_name' = %s
            LIMIT 1;
        """, (class_name, subject, chapter_name))
        row = cur.fetchone()
        if row:
            cur.close()
            conn.close()
            return row[0]
            
        # 2. Try match on class and subject
        cur.execute("""
            SELECT c.name
            FROM langchain_pg_collection c
            JOIN langchain_pg_embedding e ON c.uuid = e.collection_id
            WHERE e.cmetadata->>'class_name' = %s
              AND e.cmetadata->>'subject' = %s
            LIMIT 1;
        """, (class_name, subject))
        row = cur.fetchone()
        if row:
            cur.close()
            conn.close()
            return row[0]
            
        # 3. Try match on class only
        cur.execute("""
            SELECT c.name
            FROM langchain_pg_collection c
            JOIN langchain_pg_embedding e ON c.uuid = e.collection_id
            WHERE e.cmetadata->>'class_name' = %s
            LIMIT 1;
        """, (class_name,))
        row = cur.fetchone()
        if row:
            cur.close()
            conn.close()
            return row[0]
            
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error finding collection dynamically: {e}")
        
    return COLLECTION_NAME

