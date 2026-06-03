import os
import re
import tempfile
import logging
from fastapi import APIRouter, File, UploadFile, HTTPException, Form, Depends
from pydantic import BaseModel

# LangChain Imports
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import PGVector

# Import the lazy-loaded vector store and helpers from database
from database import get_vector_store, get_collection_for_input

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Ingestion"])

# --- Pydantic Models ---
class IngestionResponse(BaseModel):
    message: str
    chunks_processed: int
    metadata: dict
    route_used: str

# --- KRUTIDEV TO UNICODE CONVERTER (The Data Science Fix) ---
def convert_krutidev_to_unicode(text: str) -> str:
    """
    Cleans legacy KrutiDev Hindi/Sanskrit font encodings and maps them to standard Unicode.
    """
    if not text:
        return ""

    # Step 1: Handle 'f' (chhoti ee matra 'ि')
    text = re.sub(r'f(.)', r'\1f', text)

    # Step 2: Handle 'Z' (reph 'र्')
    text = re.sub(r'(.)Z', r'Z\1', text)

    # Step 3: Core KrutiDev Character Mapping
    replacements = {
        "oqQ": "कु", "fozQ": "क्रि", "Fk": "थ", "vk": "आ", "bZ": "ई", "b": "इ",
        "¾": "=", "ß": "त्र", "Z": "र्", "f": "ि", "k": "ा", "s": "े", "S": "ै",
        "a": "ं", "~": "्", "%": "ः", "Q": "क", "o": "व", "i": "प", "j": "र",
        "h": "ी", "r": "त", "e": "म", "y": "ल", "u": "न", "w": "ू", "x": "ग",
        "I": "प्", "A": "ाँ", "c": "ब", "C": "ब्", "d": "क", "D": "क्", "g": "ह",
        "G": "ह्", "l": "स", "L": "स्", "m": "उ", "M": "श्", "n": "द", "N": "छ",
        "p": "च", "P": "च्", "q": "ु", "R": "त्", "t": "ू", "T": "ट्", "U": "न्",
        "v": "अ", "V": "ट", "X": "ग्", "Y": "ल्", "z": "्र", "`": "़", "O": "व्",
        "K": "ज्ञ", "J": "श्र", "E": "म्", "¡": "ँ", "¯": "ं", "[": "ख", ";": "य", 
        ".k": "ण", "B": "ठ", "1": "१", "2": "२", "3": "३", "4": "४", "5": "५", 
        "6": "६", "7": "७", "8": "८", "9": "९", "0": "०", "-": ".", "(": "(", ")": ")"
    }

    sorted_replacements = sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True)

    for kruti_char, unicode_char in sorted_replacements:
        text = text.replace(kruti_char, unicode_char)

    return text

# --- CORE LOGIC (DRY Principle) ---
async def process_document(
    file: UploadFile, 
    class_name: str, 
    subject: str, 
    chapter_name: str, 
    vs: PGVector, 
    is_devanagari: bool, 
    route_name: str
):
    """ Core function to handle extraction, optional NLP cleaning, chunking, and embedding. """
    logger.info(f"[{route_name.upper()}] Starting ingestion for Class: {class_name}, Subject: {subject}, Chapter: {chapter_name}")
    
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDFs supported.")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File error: {str(e)}")

    try:
        logger.info("Parsing PDF document...")
        loader = PyMuPDFLoader(tmp_path)
        documents = loader.load()
        
        # CONDITIONALLY APPLY THE DATA SCIENCE FIX
        if is_devanagari:
            logger.info("Running NLP Regex Translation: Converting legacy KrutiDev encoding to Unicode Devanagari...")
            for doc in documents:
                doc.page_content = convert_krutidev_to_unicode(doc.page_content)
        else:
            logger.info("Bypassing translation: English text detected.")

        logger.info("Chunking document text...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(documents)

        for chunk in chunks:
            chunk.metadata.update({
                "class_name": class_name,
                "subject": subject,
                "chapter_name": chapter_name,
                "language": "devanagari" if is_devanagari else "english"
            })

        logger.info(f"Generating embeddings and inserting {len(chunks)} vectors into PostgreSQL...")
        vs.add_documents(chunks)
        logger.info("Vector insertion complete!")

    except Exception as e:
        logger.error(f"Error during document processing: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return IngestionResponse(
        message="Lesson successfully ingested and vectorized.",
        chunks_processed=len(chunks),
        metadata={"class": class_name, "subject": subject, "chapter": chapter_name},
        route_used=route_name
    )

# --- ROUTE 1: ENGLISH ---
@router.post("/ingest/en", response_model=IngestionResponse)
async def ingest_english(
    file: UploadFile = File(...),
    class_name: str = Form(...),
    subject: str = Form(...),
    chapter_name: str = Form(...),
):
    """ Uploads an English PDF. Does NOT apply the KrutiDev fix. """
    collection_name = get_collection_for_input(class_name, subject, chapter_name)
    logger.info(f"Dynamically resolved collection for Ingestion: Class {class_name}, Subject {subject} -> '{collection_name}'")
    vs = get_vector_store(collection_name)
    return await process_document(file, class_name, subject, chapter_name, vs, is_devanagari=False, route_name="english")

# --- ROUTE 2: DEVANAGARI (HINDI/SANSKRIT) ---
@router.post("/ingest/devanagari", response_model=IngestionResponse)
async def ingest_devanagari(
    file: UploadFile = File(...),
    class_name: str = Form(...),
    subject: str = Form(...),
    chapter_name: str = Form(...),
):
    """ Uploads a Hindi/Sanskrit PDF. APPLIES the KrutiDev to Unicode NLP translation. """
    collection_name = get_collection_for_input(class_name, subject, chapter_name)
    logger.info(f"Dynamically resolved collection for Ingestion: Class {class_name}, Subject {subject} -> '{collection_name}'")
    vs = get_vector_store(collection_name)
    return await process_document(file, class_name, subject, chapter_name, vs, is_devanagari=True, route_name="devanagari")