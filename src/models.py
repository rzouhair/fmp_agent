
from operator import add
from typing import Annotated, List, Optional, Dict, Any
from pydantic import BaseModel, Field


class QuestionOption(BaseModel):
    option: str

class Question(BaseModel):
    question: str = Field(description="The question text, it should be a question, not a clinical case")
    options: List[QuestionOption] = Field(description="The list of options for the question usually in alphabetical order")
    number: int = Field(description="The question number preceeding the question text")

class PageQuestionsNumbers(BaseModel):
    # total_count: int = Field(description="The total number of questions")
    question_numbers: List[int] = Field(description="The list of question numbers")
    # question_numbers_options_map: Optional[Dict[int, int]] = Field(description="The map of question numbers to the number of options they have")
    # clinical_cases_count: int = Field(description="The total number of clinical cases")
    # warning: Optional[str] = Field(description="The warning message if any")

class PageClinicalCase(BaseModel):
    clinical_case: str
    question_number: int

class PageClinicalCases(BaseModel):
    clinical_cases: List[str]

class PageQuestionsClinicalCases(BaseModel):
    clinical_cases: List[str]
    questions: List[str]

class PageQuestionsText(BaseModel):
    questions: str

class PageQuestions(BaseModel):
    questions: List[Question]

class PageFixes(BaseModel):
    fixes: List[str]

class ExtractionState(BaseModel):

    current_phase: str = "extract_page_text"

    exam_images: List[str] = []
    current_page_index: int = 0

    pages_fixes_map: Dict[str, str] = {}

    pages_text: List[str] = []
    pages_text_concatenated: str = ""
    pages_text_cleaned: str = ""

    pages_questions_map: Dict[str, PageQuestionsNumbers] = {}
    pages_questions_text_map: Dict[str, str] = {}
    pages_clinical_cases_map: Dict[str, List[str]] = {}

    # List of all extracted questions from the exam, accumulated as they are assigned
    exam_questions: List[Question] = []
    formatted_questions: List[Any] = []

