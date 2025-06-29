
from operator import add
from typing import Annotated, List, Optional, Dict, Any
from pydantic import BaseModel, Field

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

class QuestionOption(BaseModel):
    option: str = Field(description="The option text, it should be a preceeded by a letter followed by a hyphen, and the option text ands before the beginning of the next question prefix")

class Question(BaseModel):
    question: str = Field(description="The full text of the question itself, as it is with no alterations, excluding the number.")
    options: List[QuestionOption] = Field(description="A list of all possible options for the question, as they are in the text, excluding the prefix letter.")
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

class ExtractionResponse(BaseModel):
    success: bool
    total_questions: int
    clinical_cases: List[ClinicalCaseResponse]
    questions: List[QuestionResponse]
    pages_questions_map: Dict[str, PageQuestionsNumbers]
    page_numbers: List[List[int]]
    message: str = ""


class PageData(BaseModel):
    questions_count: int
    is_instructions_page: bool
    is_corrections_table_page: bool

class PageDataOutput(BaseModel):
    data: List[PageData]

class DocumentClinicalCase(BaseModel):
    clinical_case: str = Field(description="The clinical case text")
    start_question_number: int = Field(description="The start question number of the clinical case")
    end_question_number: int = Field(description="The end question number of the clinical case")

class DocumentClinicalCaseOutput(BaseModel):
    data: List[DocumentClinicalCase]

class DocumentExtractionState(BaseModel):

    document_contents: List[str] = []

    pdf_path: str = ""
    current_page_index: int = 0

    exam_images: List[str] = []

    pages_data: PageDataOutput = PageDataOutput(data=[])
    questions: List[Question] = []
    questions_raw_text: str = ""
    questions_markdown_text: str = ""

    pages_clinical_cases: List[ClinicalCaseResponse] = []
    output_clinical_cases: List[ClinicalCaseResponse] = []

    exam_questions: List[Question] = []
    formatted_questions: List[Any] = []

