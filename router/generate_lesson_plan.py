import os
import logging
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List

# LangChain Imports
from langchain_community.vectorstores import PGVector
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

# Import the lazy-loaded vector store and helpers from database
from database import get_vector_store, get_collection_for_input

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize the router
router = APIRouter(prefix="/api/v1", tags=["Lesson Plan Generation"])

# --- Pydantic Models for Structured AI Output ---
class LearningObjective(BaseModel):
    objective: str = Field(description="A clear, measurable learning outcome")

class LessonActivity(BaseModel):
    duration: str = Field(description="Estimated time for this activity (e.g., '15 mins')")
    activity_title: str = Field(description="Title of the instructional activity")
    description: str = Field(description="Detailed description of teacher and student actions")

class AssessmentStrategy(BaseModel):
    assessment_type: str = Field(description="Type of assessment (e.g., Quiz, Rubric, Observation)")
    description: str = Field(description="How it measures the learning objectives and AI-driven insights on student evaluation")

class LessonPlanOutput(BaseModel):
    lesson_title: str = Field(description="Provide a catchy and relevant title for the lesson")
    curriculum_alignment: str = Field(description="How this lesson aligns with standard curriculum expectations")
    learning_objectives: List[LearningObjective] = Field(default_factory=list, description="List of learning outcomes")
    materials_needed: List[str] = Field(default_factory=list, description="List of resources, materials, or tools needed")
    introduction: str = Field(description="A hook or introduction to engage students at the start")
    activities: List[LessonActivity] = Field(default_factory=list, description="The main instructional activities")
    conclusion: str = Field(description="Wrap-up or summary of the lesson")
    assessments: List[AssessmentStrategy] = Field(default_factory=list, description="Recommended assessments with AI-driven insights")

class LessonPlanRequest(BaseModel):
    class_name: str
    subject: str
    chapter_name: str
    lesson_duration: int = 60

# --- Endpoints ---
@router.post("/generate_lesson_plan", response_model=LessonPlanOutput)
async def generate_lesson_plan(request: LessonPlanRequest):
    """ Retrieves context from Postgres and designs a comprehensive, curriculum-aligned lesson plan. """
    class_name = request.class_name
    subject = request.subject
    chapter_name = request.chapter_name
    lesson_duration = request.lesson_duration

    logger.info(f"Generating lesson plan for Class: {class_name}, Subject: {subject}, Chapter: {chapter_name}")
    logger.info(f"Target duration: {lesson_duration} minutes")

    # Dynamically resolve which collection this class/subject belongs to in PostgreSQL
    collection_name = get_collection_for_input(class_name, subject, chapter_name)
    logger.info(f"Dynamically resolved collection for Class {class_name}, Subject {subject} -> '{collection_name}'")
    vs = get_vector_store(collection_name)

    # 1. Vector Search (RAG)
    try:
        retriever = vs.as_retriever(
            search_kwargs={
                "k": 25, 
                "filter": {
                    "class_name": class_name,
                    "subject": subject,
                    "chapter_name": chapter_name
                }
            }
        )

        docs = retriever.invoke(f"Extract key concepts for {chapter_name} to build a lesson plan")
        context_text = "\n\n".join([doc.page_content for doc in docs])
        
        logger.info(f"Successfully retrieved {len(docs)} chunks from the database.")
    except Exception as e:
        logger.error(f"Database retrieval failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Database retrieval failed.")

    # Check for empty context outside the try-except block so the 404 propagates cleanly
    if not context_text:
        logger.warning("No context found in PostgreSQL for the given filters.")
        raise HTTPException(status_code=404, detail="No ingested data found for this class and chapter.")

    # 2. Initialize the Standard LLM securely using .env variables
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
    if not deepseek_api_key or "your-deepseek-api-key" in deepseek_api_key:
        logger.error("DEEPSEEK_API_KEY is missing or invalid.")
        raise HTTPException(status_code=500, detail="CRITICAL: DEEPSEEK_API_KEY is missing or invalid in .env file.")
    
    llm = ChatOpenAI(
        model="deepseek-chat", 
        api_key=deepseek_api_key, 
        base_url="https://api.deepseek.com",
        temperature=0.2
    )

    # 3. Setup the Pydantic Output Parser
    parser = PydanticOutputParser(pydantic_object=LessonPlanOutput)

    # 4. Create the Prompt with Format Instructions injected
    system_prompt = """
    You are an expert curriculum designer and educator. Your task is to design a comprehensive, 
    curriculum-aligned lesson plan in seconds with AI-driven insights based strictly on the textbook context provided below.
    Do not use outside knowledge. If the answer isn't in the context, do your best based only on the text.
    
    REQUIREMENTS:
    - Design a lesson plan that spans approximately {lesson_duration} minutes.
    - Ensure activities are engaging, well-structured, and clearly detailed.
    - Include explicit learning objectives and curriculum alignment.
    - Provide AI-driven assessment strategies to measure student success.

    MATHEMATICAL NOTATION:
    - Use LaTeX notation for ALL mathematical expressions without exception.
    - Wrap inline math with single dollar signs: $...$ (e.g., $\sqrt{{48x(x+14)}}$).
    - Wrap block/display equations with double dollar signs: $$...$$ on their own line.
    - Never use plain-text alternatives like √, ×, ÷, or superscript notation outside LaTeX.
    
    {format_instructions}
    
    Context from Chapter:
    {context}
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Design the lesson plan for Class {class_name}, Subject: {subject}, Chapter: {chapter_name}.")
    ])

    # 5. Execute the Chain (Prompt -> LLM -> Parser)
    logger.info("Sending prompt to DeepSeek API...")
    chain = prompt | llm | parser
    
    try:
        result = chain.invoke({
            "context": context_text,
            "class_name": class_name,
            "subject": subject,
            "chapter_name": chapter_name,
            "lesson_duration": lesson_duration,
            "format_instructions": parser.get_format_instructions()
        })
        logger.info("Successfully generated and parsed JSON from DeepSeek.")
        return result
    except Exception as e:
        logger.error(f"LLM Generation or Parsing failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"LLM Generation failed: {str(e)}")