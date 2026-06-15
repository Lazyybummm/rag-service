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

router = APIRouter(prefix="/api/v1", tags=["Study Notes Generation"])

class StudySummary(BaseModel):
    heading: str = Field(description="Heading for this section of notes")
    bullet_points: List[str] = Field(description="Synthesized bullet summaries of the complex lecture/text")

class StudyNotesOutput(BaseModel):
    topic: str = Field(description="The topic of the study notes")
    overview: str = Field(description="A brief overview of the topic")
    sections: List[StudySummary] = Field(default_factory=list, description="Sections containing bulleted summaries")
    key_terms: List[str] = Field(default_factory=list, description="List of important vocabulary or key terms")

class StudyNotesRequest(BaseModel):
    class_name: str
    subject: str
    chapter_name: str
    topic: str

@router.post("/generate_study_notes", response_model=StudyNotesOutput)
async def generate_study_notes(request: StudyNotesRequest):
    """ Retrieves context and synthesizes complex lectures into bullet summaries. """
    logger.info(f"Generating study notes for {request.class_name}, {request.subject}, {request.chapter_name}, Topic: {request.topic}")
    
    collection_name = get_collection_for_input(request.class_name, request.subject, request.chapter_name)
    vs = get_vector_store(collection_name)

    try:
        retriever = vs.as_retriever(search_kwargs={"k": 25, "filter": {"class_name": request.class_name, "subject": request.subject, "chapter_name": request.chapter_name}})
        docs = retriever.invoke(f"Extract detailed explanations for {request.topic}")
        context_text = "\n\n".join([doc.page_content for doc in docs])
    except Exception as e:
        logger.error(f"Database retrieval failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Database retrieval failed.")

    if not context_text:
        raise HTTPException(status_code=404, detail="No ingested data found for this class and chapter.")

    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
    if not deepseek_api_key or "your-deepseek-api-key" in deepseek_api_key:
        raise HTTPException(status_code=500, detail="DEEPSEEK_API_KEY is missing or invalid.")
    
    llm = ChatOpenAI(model="deepseek-chat", api_key=deepseek_api_key, base_url="https://api.deepseek.com", temperature=0.2)
    parser = PydanticOutputParser(pydantic_object=StudyNotesOutput)

    system_prompt = """
    You are an expert academic summarizer. Synthesize the provided complex lecture/textbook context into elegant, easy-to-read bullet summaries based on the provided topic.
    Do not use outside knowledge.
    
    REQUIREMENTS:
    - Break down the topic "{topic}" into logical sections.
    - Provide an overview and a list of key terms.
    - Use clear and concise bullet points for the notes.

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
        ("human", "Generate study notes for Class {class_name}, Subject: {subject}, Chapter: {chapter_name}, Topic: {topic}.")
    ])

    chain = prompt | llm | parser
    try:
        return chain.invoke({
            "context": context_text,
            "class_name": request.class_name,
            "subject": request.subject,
            "chapter_name": request.chapter_name,
            "topic": request.topic,
            "format_instructions": parser.get_format_instructions()
        })
    except Exception as e:
        logger.error(f"LLM Generation failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"LLM Generation failed: {str(e)}")
