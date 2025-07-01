import base64
import os
from jinja2 import Environment, FileSystemLoader

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from src.models import Document

# Set up the Jinja2 environment to load templates from the 'templates' directory
template_dir = os.path.join(os.path.dirname(__file__), "templates")
env = Environment(loader=FileSystemLoader(template_dir))


def _load_prompt(template_name: str) -> str:
    """Loads a prompt from a Jinja2 template."""
    return env.get_template(f"{template_name}.j2").render()


# Load the prompt content from the template file
EXTRACT_ALL_FROM_DOCUMENT_PROMPT = _load_prompt("extract_all_from_document")


load_dotenv()


class GeminiWorkflow:
    def __init__(self):
        self._model = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            temperature=0,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            # The `with_structured_output` method handles JSON formatting,
            # so `response_mime_type` is not needed here and causes a conflict.
        )
        # This forces the model to not only output JSON but to conform to the Pydantic schema of the Document class.
        self._structured_model = self._model.with_structured_output(
            schema=Document,
            include_raw=False,
        )

    def _extract_all_questions_step(self, pdf_base64_string: str) -> Document:
        """
        Takes a base64 encoded PDF and extracts all questions from it in a single call.
        """
        print("🚀 Executing PDF native extraction strategy...")
        message = HumanMessage(
            content=[
                {"type": "text", "text": EXTRACT_ALL_FROM_DOCUMENT_PROMPT},
                {
                    "type": "media",
                    "data": pdf_base64_string,
                    "mime_type": "application/pdf",
                },
            ]
        )
        
        print("📄 Sending the entire PDF to the model in a single request...")
        document = self._structured_model.invoke([message])
        
        if document and document.questions:
            print(f"✅ Successfully extracted {len(document.questions)} questions.")
        else:
            print("⚠️ Model did not return any questions.")

        return document

    def run(self, pdf_input: str, from_local_file: bool = False) -> Document:
        """
        The main entry point for the workflow.
        It can accept either a path to a local PDF file or a base64 encoded string of a PDF.
        """
        pdf_base64_string = pdf_input
        if from_local_file:
            print(f"📦 Loading PDF from local file: {pdf_input}")
            with open(pdf_input, "rb") as f:
                pdf_base64_string = base64.b64encode(f.read()).decode("utf-8")
        else:
            # The input is already a base64 string.
            print("📦 Received PDF as base64 string.")

        document = self._extract_all_questions_step(pdf_base64_string)
        return document
