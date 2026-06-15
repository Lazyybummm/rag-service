import os
import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List

from langchain_community.vectorstores import PGVector
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from database import get_vector_store, get_collection_for_input

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Presentation Outline Generation"])

class PresentationSlide(BaseModel):
    slide_number: int = Field(description="The sequential number of the slide")
    title: str = Field(description="Title of the slide")
    bullet_points: List[str] = Field(description="Key points to be shown on the slide")
    speaker_notes: str = Field(description="Detailed narrative script or notes for the presenter")

class PresentationOutput(BaseModel):
    presentation_title: str = Field(description="Overall title of the presentation")
    slides: List[PresentationSlide] = Field(default_factory=list, description="Slide by slide narrative arc")

class PresentationRequest(BaseModel):
    class_name: str
    subject: str
    chapter_name: str
    topic: str
    num_slides: int = 7

@router.post("/generate_presentation_outline", response_model=PresentationOutput)
async def generate_presentation_outline(request: PresentationRequest):
    """ Retrieves context and develops a slide-by-slide narrative arc. """
    logger.info(f"Generating presentation outline for {request.class_name}, {request.subject}, {request.chapter_name}, Topic: {request.topic}")
    
    collection_name = get_collection_for_input(request.class_name, request.subject, request.chapter_name)
    vs = get_vector_store(collection_name)

    try:
        retriever = vs.as_retriever(search_kwargs={"k": 25, "filter": {"class_name": request.class_name, "subject": request.subject, "chapter_name": request.chapter_name}})
        docs = retriever.invoke(f"Extract key concepts to create a presentation for {request.topic}")
        context_text = "\n\n".join([doc.page_content for doc in docs])
    except Exception as e:
        logger.error(f"Database retrieval failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Database retrieval failed.")

    if not context_text:
        raise HTTPException(status_code=404, detail="No ingested data found for this class and chapter.")

    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
    if not deepseek_api_key or "your-deepseek-api-key" in deepseek_api_key:
        raise HTTPException(status_code=500, detail="DEEPSEEK_API_KEY is missing or invalid.")
    
    llm = ChatOpenAI(model="deepseek-chat", api_key=deepseek_api_key, base_url="https://api.deepseek.com", temperature=0.3)
    parser = PydanticOutputParser(pydantic_object=PresentationOutput)

    system_prompt = """
    You are an expert instructional designer. Develop a presentation outline with a compelling slide-by-slide narrative arc based strictly on the textbook context below.
    Do not use outside knowledge.
    
    REQUIREMENTS:
    - Generate EXACTLY {num_slides} slides for the topic "{topic}".
    - Each slide must have a title, bullet points, and speaker notes that guide the presenter.
    - Include an intro slide, body slides, and a conclusion slide.

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
        ("human", "Generate the presentation outline for Class {class_name}, Subject: {subject}, Chapter: {chapter_name}, Topic: {topic}.")
    ])

    chain = prompt | llm | parser
    try:
        return chain.invoke({
            "context": context_text,
            "class_name": request.class_name,
            "subject": request.subject,
            "chapter_name": request.chapter_name,
            "topic": request.topic,
            "num_slides": request.num_slides,
            "format_instructions": parser.get_format_instructions()
        })
    except Exception as e:
        logger.error(f"LLM Generation failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"LLM Generation failed: {str(e)}")
