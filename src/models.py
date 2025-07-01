from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Optional


# Core workflow models
class Option(BaseModel):
    option: str = Field(..., description="The text of the option")


class Question(BaseModel):
    question_number: int = Field(..., description="The number of the question")
    page_number: int = Field(..., description="The page number where the question starts")
    clinical_case: Optional[str] = Field(
        None, description="The clinical case associated with the question, if any"
    )
    question: str = Field(..., description="The full text of the question")
    options: List[Option] = Field(
        ..., description="A list of all options for the question"
    )


class Document(BaseModel):
    questions: List[Question] = Field(
        ..., description="A list of all questions found in the document"
    )


# API Response Models
class OptionResponse(BaseModel):
    option: str
    isCorrect: bool
    justification: str
    images: List[str]


class QuestionResponse(BaseModel):
    questionString: str
    explanation: str
    tag: str
    options: List[OptionResponse]


class ClinicalCaseResponse(BaseModel):
    question_numbers: List[int]
    clinical_case: str
    type: str
    images: List[str]


class ExtractionResponse(BaseModel):
    success: bool
    total_questions: int
    questions: List[QuestionResponse]
    clinical_cases: List[ClinicalCaseResponse]
    pages_questions_map: dict
    page_numbers: List[List[int]]
    message: str


# --- Old models ---
# These are likely deprecated but are kept for now to prevent import errors in other parts of the code
# that may not have been fully refactored yet.
class PageQuestions(BaseModel):
    questions: List[Question] = Field(..., description="A list of all questions")


class DocumentClinicalCase(BaseModel):
    clinical_case: str = Field(description="The clinical case text")
    start_question_number: int = Field(description="The start question number of the clinical case")
    end_question_number: int = Field(description="The end question number of the clinical case")
    is_corrections_table_page: bool = Field(False, description="Whether the page contains a corrections table")


class DocumentExtractionState(BaseModel):
    """A simple state model for the one-shot extraction workflow."""
    pdf_path: str = Field(None, description="Path to the PDF file")
    exam_images: List[str] = Field(default_factory=list, description="Base64 encoded images of the exam pages")
    
    questions: List[Question] = Field(default_factory=list, description="The final, extracted questions")
    pages_clinical_cases: List[DocumentClinicalCase] = Field(default_factory=list, description="The extracted clinical cases")

