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

router = APIRouter(prefix="/api/v1", tags=["Quiz Generation"])

class QuizMCQ(BaseModel):
    question: str = Field(description="The multiple choice question")
    options: List[str] = Field(description="Exactly 4 options")
    correct_answer: str = Field(description="The correct option")

class QuizShortAnswer(BaseModel):
    question: str = Field(description="A short answer question based on the topic")
    answer_key: str = Field(description="The correct answer")

class QuizOutput(BaseModel):
    title: str = Field(description="Title of the quiz")
    mcqs: List[QuizMCQ] = Field(default_factory=list, description="A list of instant MCQs")
    short_answers: List[QuizShortAnswer] = Field(default_factory=list, description="A list of short answers")

class QuizRequest(BaseModel):
    class_name: str
    subject: str
    chapter_name: str
    topic: str
    num_mcqs: int = 5
    num_short_answers: int = 5

@router.post("/generate_quiz", response_model=QuizOutput)
async def generate_quiz(request: QuizRequest):
    """ Retrieves context and generates an instant quiz. """
    logger.info(f"Generating quiz for {request.class_name}, {request.subject}, {request.chapter_name}, Topic: {request.topic}")
    
    collection_name = get_collection_for_input(request.class_name, request.subject, request.chapter_name)
    vs = get_vector_store(collection_name)

    try:
        retriever = vs.as_retriever(search_kwargs={"k": 25, "filter": {"class_name": request.class_name, "subject": request.subject, "chapter_name": request.chapter_name}})
        docs = retriever.invoke(f"Extract key facts and details for {request.topic}")
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
    parser = PydanticOutputParser(pydantic_object=QuizOutput)

    system_prompt = """
    You are an expert teacher. Generate an instant quiz based strictly on the textbook context below. 
    Do not use outside knowledge.
    
    REQUIREMENTS:
    - Generate EXACTLY {num_mcqs} Multiple Choice Questions (MCQs) for the topic "{topic}".
    - Generate EXACTLY {num_short_answers} Short Answer Questions for the topic "{topic}".

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
        ("human", "Generate the quiz for Class {class_name}, Subject: {subject}, Chapter: {chapter_name}, Topic: {topic}.")
    ])

    chain = prompt | llm | parser
    try:
        return chain.invoke({
            "context": context_text,
            "class_name": request.class_name,
            "subject": request.subject,
            "chapter_name": request.chapter_name,
            "topic": request.topic,
            "num_mcqs": request.num_mcqs,
            "num_short_answers": request.num_short_answers,
            "format_instructions": parser.get_format_instructions()
        })
    except Exception as e:
        logger.error(f"LLM Generation failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"LLM Generation failed: {str(e)}")
