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

router = APIRouter(prefix="/api/v1", tags=["Question Paper Generation"])

class PaperQuestion(BaseModel):
    question_text: str = Field(description="The text of the question")
    marks: int = Field(description="Marks allocated for this question")
    answer_key: str = Field(description="The automated answer key or rubric for grading this question")

class PaperSection(BaseModel):
    section_name: str = Field(description="Name of the section (e.g., Section A: One Mark Questions)")
    instructions: str = Field(description="Instructions for this section")
    total_section_marks: int = Field(description="Total marks for this section")
    questions: List[PaperQuestion] = Field(default_factory=list, description="List of questions in this section")

class QuestionPaperOutput(BaseModel):
    exam_title: str = Field(description="Title of the examination paper")
    total_marks: int = Field(description="Total marks for the paper")
    duration: str = Field(description="Suggested duration for the exam (e.g., '2 Hours')")
    sections: List[PaperSection] = Field(default_factory=list, description="Sections of the question paper")

class QuestionPaperRequest(BaseModel):
    class_name: str
    subject: str
    chapter_name: str
    total_marks: int = 50

@router.post("/generate_question_paper", response_model=QuestionPaperOutput)
async def generate_question_paper(request: QuestionPaperRequest):
    """ Retrieves context and generates a formal exam question paper. """
    logger.info(f"Generating question paper for {request.class_name}, {request.subject}, {request.chapter_name}, Marks: {request.total_marks}")
    
    collection_name = get_collection_for_input(request.class_name, request.subject, request.chapter_name)
    vs = get_vector_store(collection_name)

    try:
        retriever = vs.as_retriever(search_kwargs={"k": 30, "filter": {"class_name": request.class_name, "subject": request.subject, "chapter_name": request.chapter_name}})
        docs = retriever.invoke(f"Extract key concepts for {request.chapter_name} formal exam")
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
    parser = PydanticOutputParser(pydantic_object=QuestionPaperOutput)

    system_prompt = """
    You are an expert examiner. Design a formal term exam question paper with automated marks allocations and answer keys, based strictly on the textbook context below.
    Do not use outside knowledge.
    
    REQUIREMENTS:
    - The exam should sum up exactly to {total_marks} marks total across all sections.
    - Organize the paper into logical sections (e.g., Objective, Short Answer, Long Answer).
    - Provide an automated answer key mapping for each question.

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
        ("human", "Generate the exam paper for Class {class_name}, Subject: {subject}, Chapter: {chapter_name}.")
    ])

    chain = prompt | llm | parser
    try:
        return chain.invoke({
            "context": context_text,
            "class_name": request.class_name,
            "subject": request.subject,
            "chapter_name": request.chapter_name,
            "total_marks": request.total_marks,
            "format_instructions": parser.get_format_instructions()
        })
    except Exception as e:
        logger.error(f"LLM Generation failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"LLM Generation failed: {str(e)}")
