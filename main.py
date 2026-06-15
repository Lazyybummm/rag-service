import os
import warnings
from fastapi import FastAPI
import uvicorn
from dotenv import load_dotenv
from router.export import router as export_router

# Load environment variables from the .env file BEFORE doing anything else
load_dotenv()

# Suppress the Python 3.14 Pydantic V1 warning
warnings.filterwarnings("ignore", message=".*Core Pydantic V1.*")

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Lesson Plan RAG System")

# Configure CORS Middleware to allow cross-origin requests from standard frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Router Registration ---
# We import these AFTER defining get_vector_store to prevent circular imports.
from router.ingest import router as ingest_router
from router.generate_lesson_plan import router as generate_router
from router.generate_worksheet import router as worksheet_router
from router.generate_quiz import router as quiz_router
from router.generate_question_paper import router as question_paper_router
from router.generate_study_notes import router as study_notes_router
from router.generate_presentation_outline import router as presentation_router
from router.generate_rubric import router as rubric_router

# Register the routes with the main app
app.include_router(ingest_router)
app.include_router(generate_router)
app.include_router(worksheet_router)
app.include_router(quiz_router)
app.include_router(question_paper_router)
app.include_router(study_notes_router)
app.include_router(presentation_router)
app.include_router(rubric_router)
app.include_router(export_router)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8001))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)