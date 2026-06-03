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

router = APIRouter(prefix="/api/v1", tags=["Worksheet Generation"])

class WorksheetQuestion(BaseModel):
    question: str = Field(description="The practice question")
    difficulty: str = Field(description="Difficulty level: Easy, Medium, or Hard")
    answer_key: str = Field(description="The ideal correct answer or solution steps")

class WorksheetOutput(BaseModel):
    title: str = Field(description="Title of the worksheet")
    questions: List[WorksheetQuestion] = Field(default_factory=list, description="A list of practice questions with varying difficulties")

class WorksheetRequest(BaseModel):
    class_name: str
    subject: str
    chapter_name: str
    topic: str
    num_questions: int = 10

@router.post("/generate_worksheet", response_model=WorksheetOutput)
async def generate_worksheet(request: WorksheetRequest):
    """ Retrieves context and generates a topic-specific worksheet with varying difficulty. """
    logger.info(f"Generating worksheet for {request.class_name}, {request.subject}, {request.chapter_name}, Topic: {request.topic}")
    
    collection_name = get_collection_for_input(request.class_name, request.subject, request.chapter_name)
    vs = get_vector_store(collection_name)

    try:
        retriever = vs.as_retriever(search_kwargs={"k": 25, "filter": {"class_name": request.class_name, "subject": request.subject, "chapter_name": request.chapter_name}})
        docs = retriever.invoke(f"Extract key concepts for {request.topic}")
        context_text = "\n\n".join([doc.page_content for doc in docs])
    except Exception as e:
        logger.error(f"Database retrieval failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Database retrieval failed.")

    if not context_text:
        raise HTTPException(status_code=404, detail="No ingested data found for this class and chapter.")

    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
    if not deepseek_api_key or "your-deepseek-api-key" in deepseek_api_key:
        raise HTTPException(status_code=500, detail="DEEPSEEK_API_KEY is missing or invalid in .env file.")
    
    llm = ChatOpenAI(model="deepseek-chat", api_key=deepseek_api_key, base_url="https://api.deepseek.com", temperature=0.2)
    parser = PydanticOutputParser(pydantic_object=WorksheetOutput)

    system_prompt = """
    You are an expert teacher. Generate a topic-specific practice sheet with varying difficulty levels 
    based strictly on the textbook context below. Do not use outside knowledge.
    
    REQUIREMENTS:
    - Generate EXACTLY {num_questions} questions for the topic: {topic}.
    - Ensure a mix of Easy, Medium, and Hard difficulty levels.
    
    {format_instructions}
    
    Context from Chapter:
    {context}
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Generate the worksheet for Class {class_name}, Subject: {subject}, Chapter: {chapter_name}, Topic: {topic}.")
    ])

    chain = prompt | llm | parser
    try:
        return chain.invoke({
            "context": context_text,
            "class_name": request.class_name,
            "subject": request.subject,
            "chapter_name": request.chapter_name,
            "topic": request.topic,
            "num_questions": request.num_questions,
            "format_instructions": parser.get_format_instructions()
        })
    except Exception as e:
        logger.error(f"LLM Generation failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"LLM Generation failed: {str(e)}")
