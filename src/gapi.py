import tempfile
import os
import traceback
import base64
from typing import List, Dict
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field

from src.api import ExtractionResponse, OptionResponse, QuestionResponse
from src.gemini_workflow import GeminiWorkflow

from .models import ClinicalCaseResponse, DocumentExtractionState, PageQuestionsNumbers


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
        workflow = GeminiWorkflow()
        
        # Run the workflow with the temporary PDF file
        final_state: DocumentExtractionState = await workflow.run(pdf_path=temp_file_path)

        print(f"Final State: {final_state.pages_clinical_cases}")

        questions_response: List[QuestionResponse] = []

        clinical_cases_response: List[ClinicalCaseResponse] = []
        for clinical_case in final_state.pages_clinical_cases:
            clinical_cases_response.append(ClinicalCaseResponse(
                question_numbers=clinical_case.question_numbers,
                clinical_case=clinical_case.clinical_case,
                type="clinicalCase",
                images=[]
            ))

        import re
        for question in final_state.questions:
            question_response = QuestionResponse(
                questionString=question.question,
                explanation="",
                tag="",
                options=[
                    OptionResponse(
                        option=re.sub(r"^[A-Z](?:[\)\-\\\.]| -|\. |- )\s*", "", option.option.strip()),
                        isCorrect=False,
                        justification="",
                        images=[]
                    )
                    for option in question.options
                ]
            )
            questions_response.append(question_response)

        page_numbers: List[List[int]] = []
        for page_index, page_data_item in enumerate(final_state.pages_data.data):
            questions_sequence = list(range(page_data_item.start_question_number - 1, page_data_item.end_question_number))

            if page_data_item.is_instructions_page or page_data_item.is_corrections_table_page or len(questions_sequence) > 9 or questions_sequence[0] < 0 :
              page_numbers.append([])

            else:
              page_numbers.append(questions_sequence)


        pages_questions_map: Dict[str, PageQuestionsNumbers] = {}

        print(ExtractionResponse(
            success=True,
            total_questions=len(questions_response),
            questions=questions_response,
            clinical_cases=clinical_cases_response,
            pages_questions_map=pages_questions_map,
            page_numbers=page_numbers,
            message=f"Successfully extracted {len(questions_response)} questions from {filename}"
        ))

        return ExtractionResponse(
            success=True,
            total_questions=len(questions_response),
            questions=questions_response,
            clinical_cases=clinical_cases_response,
            pages_questions_map=pages_questions_map,
            page_numbers=page_numbers,
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