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

router = APIRouter(prefix="/api/v1", tags=["Rubric Generation"])

class GradingCriteria(BaseModel):
    criterion_name: str = Field(description="The specific skill or attribute being assessed (e.g., Concept Accuracy, Clarity)")
    weight: int = Field(description="Percentage or points assigned to this criterion")
    excellent: str = Field(description="Description of excellent performance (full marks)")
    good: str = Field(description="Description of good performance (partial marks)")
    needs_improvement: str = Field(description="Description of performance needing improvement (low marks)")
    poor: str = Field(description="Description of poor performance (zero/minimal marks)")

class RubricOutput(BaseModel):
    assignment_title: str = Field(description="Title of the assignment or task")
    total_score: int = Field(description="Maximum total score achievable based on the criteria")
    criteria: List[GradingCriteria] = Field(default_factory=list, description="Objective grading criteria")

class RubricRequest(BaseModel):
    class_name: str
    subject: str
    chapter_name: str
    assignment_description: str
    total_score: int = 100

@router.post("/generate_rubric", response_model=RubricOutput)
async def generate_rubric(request: RubricRequest):
    """ Creates objective grading criteria for a given assignment based on chapter context. """
    logger.info(f"Generating rubric for {request.class_name}, {request.subject}, {request.chapter_name}, Assignment: {request.assignment_description}")
    
    collection_name = get_collection_for_input(request.class_name, request.subject, request.chapter_name)
    vs = get_vector_store(collection_name)

    try:
        retriever = vs.as_retriever(search_kwargs={"k": 20, "filter": {"class_name": request.class_name, "subject": request.subject, "chapter_name": request.chapter_name}})
        docs = retriever.invoke(f"Extract key concepts for rubric evaluation regarding {request.assignment_description}")
        context_text = "\n\n".join([doc.page_content for doc in docs])
    except Exception as e:
        logger.error(f"Database retrieval failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Database retrieval failed.")

    if not context_text:
        raise HTTPException(status_code=404, detail="No ingested data found for this class and chapter.")

    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
    if not deepseek_api_key or "your-deepseek-api-key" in deepseek_api_key:
        raise HTTPException(status_code=500, detail="DEEPSEEK_API_KEY is missing or invalid.")
    
    llm = ChatOpenAI(model="deepseek-chat", api_key=deepseek_api_key, base_url="https://api.deepseek.com", temperature=0.1)
    parser = PydanticOutputParser(pydantic_object=RubricOutput)

    system_prompt = """
    You are an expert evaluator. Create an objective grading rubric for the assignment described below, 
    tailored to the educational concepts from the textbook context. Do not use outside knowledge.
    
    REQUIREMENTS:
    - Assess the assignment: "{assignment_description}".
    - Provide distinct criteria (e.g., Content, Structure, Critical Thinking) summing up to {total_score}.
    - For each criterion, describe what constitutes Excellent, Good, Needs Improvement, and Poor.
    
    {format_instructions}
    
    Context from Chapter:
    {context}
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Generate the rubric for Class {class_name}, Subject: {subject}, Chapter: {chapter_name}, Assignment: {assignment_description}.")
    ])

    chain = prompt | llm | parser
    try:
        return chain.invoke({
            "context": context_text,
            "class_name": request.class_name,
            "subject": request.subject,
            "chapter_name": request.chapter_name,
            "assignment_description": request.assignment_description,
            "total_score": request.total_score,
            "format_instructions": parser.get_format_instructions()
        })
    except Exception as e:
        logger.error(f"LLM Generation failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"LLM Generation failed: {str(e)}")
