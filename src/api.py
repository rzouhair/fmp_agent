import base64
from typing import List
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field

from src.models import (
    ClinicalCaseResponse,
    ExtractionResponse,
    OptionResponse,
    QuestionResponse,
    Question,
    Document,
)
from .gemini_workflow import GeminiWorkflow


class Base64FileRequest(BaseModel):
    base64: str = Field(..., description="Base64 encoded PDF file data")


app = FastAPI(
    title="PDF Question Extractor API",
    description="Extract multiple choice questions from PDF exam files",
    version="2.0.0",  # Version bump to reflect new implementation
)


@app.post("/extract-questions", response_model=ExtractionResponse)
def extract_questions_from_pdf(
    request: Base64FileRequest = Body(..., description="Base64 encoded PDF file data")
) -> ExtractionResponse:
    """
    Extract multiple choice questions from a PDF file using Gemini 1.5 Pro.
    
    The API accepts base64 encoded PDF data and will:
    1. Send the PDF directly to the AI model.
    2. The model analyzes the entire document, including text and layout.
    3. Return structured JSON with all questions.
    
    Args:
        request: Base64 encoded PDF data (JSON body)
        
    Returns:
        ExtractionResponse: Structured response with extracted questions
    """
    try:
        # 1. Initialize the new workflow
        workflow = GeminiWorkflow()

        # 2. Run the workflow directly with the base64 string
        # This is now a synchronous call
        document: Document = workflow.run(pdf_input=request.base64, from_local_file=False)

        # 3. Map the results to the API response models
        questions_response: List[QuestionResponse] = []
        clinical_cases_response: List[ClinicalCaseResponse] = []
        
        # Keep track of clinical cases to avoid duplicates
        processed_clinical_cases = set()

        for question in document.questions:
            # Map question and options
            question_response = QuestionResponse(
                questionString=question.question,
                explanation="",
                tag="",
                options=[
                    OptionResponse(option=opt.option, isCorrect=False, justification="", images=[])
                    for opt in question.options
                ],
            )
            questions_response.append(question_response)

            # Map clinical case if it exists and hasn't been processed yet
            if question.clinical_case and question.clinical_case not in processed_clinical_cases:
                cc_resp = ClinicalCaseResponse(
                    question_numbers=[q.question_number for q in document.questions if q.clinical_case == question.clinical_case],
                    clinical_case=question.clinical_case,
                    type="clinicalCase",
                    images=[],
                )
                clinical_cases_response.append(cc_resp)
                processed_clinical_cases.add(question.clinical_case)

        return ExtractionResponse(
            success=True,
            total_questions=len(questions_response),
            questions=questions_response,
            clinical_cases=clinical_cases_response,
            pages_questions_map={},  # This is deprecated by the new model
            page_numbers=[],  # This is deprecated by the new model
            message=f"Successfully extracted {len(questions_response)} questions."
        )

    except Exception as e:
        # The new workflow already prints traceback, so we can keep this simpler
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "PDF Question Extractor API is running"}


@app.get("/")
def root():
    """Root endpoint with API information"""
    return {
        "message": "PDF Question Extractor API",
        "version": "2.0.0",
        "endpoints": {
            "POST /extract-questions": "Extract questions from PDF file (base64 only)",
            "GET /health": "Health check",
            "GET /docs": "API documentation",
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 