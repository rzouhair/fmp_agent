import tempfile
import os
import traceback
import base64
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .workflow import Workflow
from .models import PageClinicalCase, PageQuestionsNumbers, Question, ExtractionState
from .utils.pdf import extract_pdf_pages_as_images


class OptionResponse(BaseModel):
    option: str
    isCorrect: bool = False
    justification: str = ""
    images: List[str] = []

class QuestionResponse(BaseModel):
    questionString: str
    explanation: str = ""
    tag: str = None
    options: List[OptionResponse]


class ClinicalCaseResponse(BaseModel):
    question_numbers: List[int] = []
    clinical_case: str
    type: str = "clinicalCase"
    images: List[str] = []

class ExtractionResponse(BaseModel):
    success: bool
    total_questions: int
    clinical_cases: List[ClinicalCaseResponse]
    questions: List[QuestionResponse]
    pages_questions_map: Dict[str, PageQuestionsNumbers]
    page_numbers: List[List[int]]
    message: str = ""


class Base64FileRequest(BaseModel):
    base64: str = Field(..., description="Base64 encoded PDF file data")


app = FastAPI(
    title="PDF Question Extractor API",
    description="Extract multiple choice questions from PDF exam files",
    version="1.0.0"
)


@app.post("/extract-questions", response_model=ExtractionResponse)
async def extract_questions_from_pdf(
    request: Base64FileRequest = Body(..., description="Base64 encoded PDF file data")
) -> ExtractionResponse:
    """
    Extract multiple choice questions from a PDF file.
    
    The API accepts base64 encoded PDF data and will:
    1. Decode the base64 PDF data
    2. Convert PDF pages to images
    3. Use AI to extract questions and options
    4. Return structured JSON with all questions
    
    Args:
        request: Base64 encoded PDF data (JSON body)
        
    Returns:
        ExtractionResponse: Structured response with extracted questions
    """
    
    filename = "document.pdf"  # Default filename for base64 uploads
    
    try:
        # Decode base64 data
        file_data = base64.b64decode(request.base64)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid base64 data: {str(e)}"
        )
    
    # Create temporary file to store the PDF data
    temp_file_path = None
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file_path = temp_file.name
            
            # Write file data to temporary file
            temp_file.write(file_data)
            temp_file.flush()
        
        # Process the PDF through the workflow
        workflow = Workflow()
        
        # Run the workflow with the temporary PDF file
        final_state = await workflow.run(pdf_path=temp_file_path)
        
        # Convert the extracted questions to response format
        questions_response = []
        for question in final_state.exam_questions:
            question_response = QuestionResponse(
                questionString=question.question,
                explanation="",
                tag="",
                options=[OptionResponse(option=option.option, isCorrect=False, justification="", images=[]) for option in question.options]
            )
            questions_response.append(question_response)

        clinical_cases_response: List[ClinicalCaseResponse] = []
        for page_index, clinical_cases in final_state.pages_clinical_cases_map.items():
            
            if len(final_state.pages_questions_map[f"{page_index}"].question_numbers) > 0:
                first_question_number = final_state.pages_questions_map[f"{page_index}"].question_numbers[0]
            else:
                first_question_number = None
            for clinical_case_text in clinical_cases:
                cc_resp = ClinicalCaseResponse(
                    question_numbers=[first_question_number if first_question_number is not None else 0],
                    clinical_case=clinical_case_text,
                    type="clinicalCase",
                    images=[],
                )

                clinical_cases_response.append(cc_resp)

        final_page_numbers = []
        for page_index, page_numbers in final_state.pages_questions_map.items():
            final_page_numbers.append(list(map(lambda x: x - 1, page_numbers.question_numbers)))

        # Remove the temporary file after finishing the extraction
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                print(f"Warning: Could not delete temporary file {temp_file_path}: {e}")

        return ExtractionResponse(
            success=True,
            total_questions=len(questions_response),
            questions=questions_response,
            clinical_cases=clinical_cases_response,
            pages_questions_map=final_state.pages_questions_map,
            page_numbers=final_page_numbers,
            message=f"Successfully extracted {len(questions_response)} questions from {filename}"
        )
        
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Error processing PDF: {str(e)}"
        )
    
    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                print(f"Warning: Could not delete temporary file {temp_file_path}: {e}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "PDF Question Extractor API is running"}


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "PDF Question Extractor API",
        "version": "1.0.0",
        "endpoints": {
            "POST /extract-questions": "Extract questions from PDF file (base64 only)",
            "GET /health": "Health check",
            "GET /docs": "API documentation"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 