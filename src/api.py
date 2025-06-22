import tempfile
import os
import traceback
from typing import List, Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

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
    questions: List[int] = []
    text: str
    type: str = "clinicalCase"
    images: List[str] = []

class ExtractionResponse(BaseModel):
    success: bool
    total_questions: int
    clinical_cases: List[ClinicalCaseResponse]
    questions: List[QuestionResponse]
    pages_questions_map: Dict[str, PageQuestionsNumbers]
    message: str = ""


app = FastAPI(
    title="PDF Question Extractor API",
    description="Extract multiple choice questions from PDF exam files",
    version="1.0.0"
)


@app.post("/extract-questions", response_model=ExtractionResponse)
async def extract_questions_from_pdf(
    file: UploadFile = File(..., description="PDF file containing exam questions")
) -> ExtractionResponse:
    """
    Extract multiple choice questions from a PDF file.
    
    The API will:
    1. Accept a PDF file upload
    2. Convert PDF pages to images
    3. Use AI to extract questions and options
    4. Return structured JSON with all questions
    
    Args:
        file: PDF file to process
        
    Returns:
        ExtractionResponse: Structured response with extracted questions
    """
    
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400, 
            detail="File must be a PDF (.pdf extension required)"
        )
    
    # Create temporary file to store the uploaded PDF
    temp_file_path = None
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file_path = temp_file.name
            
            # Write uploaded file content to temporary file
            content = await file.read()
            temp_file.write(content)
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
            # Get the first question number for this page, if available
            if len(final_state.pages_questions_map[f"{page_index}"].question_numbers) > 0:
                first_question_number = final_state.pages_questions_map[f"{page_index}"].question_numbers[0]
            else:
                first_question_number = None
            for clinical_case_text in clinical_cases:
                cc_resp = ClinicalCaseResponse(
                    questions=[first_question_number - 1 if first_question_number is not None else 0],
                    text=clinical_case_text,
                    type="clinicalCase",
                    images=[],
                )
                # Optionally, you can add the first_question_number as an attribute if your model supports it
                # cc_resp.first_question_number = first_question_number
                clinical_cases_response.append(cc_resp)


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
            message=f"Successfully extracted {len(questions_response)} questions from {file.filename}"
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
            "POST /extract-questions": "Extract questions from PDF file",
            "GET /health": "Health check",
            "GET /docs": "API documentation"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 