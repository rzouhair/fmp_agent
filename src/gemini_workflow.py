import asyncio
import base64
import os
from typing import Dict, Any, List
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

from src.api import ClinicalCaseResponse
from src.utils.agent import render_template
from src.utils.pdf import extract_pdf_pages_as_images
from .models import DocumentClinicalCaseOutput, DocumentExtractionState, PageDataOutput, PageQuestions
from .prompts import DeveloperToolsPrompts

load_dotenv()


class GeminiWorkflow:
    def __init__(self):

        # self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-04-17", temperature=0.5, google_api_key=os.getenv("GOOGLE_API_KEY"))
        self.gLlm = self.llm
        self.gLlmCalls = 0
        self.max_quota = 10

        self.prompts = DeveloperToolsPrompts()
        self.workflow = self._build_workflow()

    def _build_workflow(self):
        graph = StateGraph(DocumentExtractionState)

        # ==================== Nodes Setup ====================

        graph.add_node("start", lambda state: state)
        graph.add_node("load_document", self._load_document_step)
        # graph.add_node("advance_page", lambda state: state)
        # graph.add_node("suggest_page_fixes", self._suggest_page_fixes_step)
        # graph.add_node("review_page_text", self._review_page_text_step)
        graph.add_node("extract_document_text", self._extract_document_text_step)
        graph.add_node("review_document_text", self._review_document_text_step)
        graph.add_node("questions_markdown_formatter", self._questions_markdown_formatter_step)

        graph.add_node("extract_document_pages_data", self._extract_document_pages_data_step)
        graph.add_node("extract_document_clinical_case", self._extract_document_clinical_case_step)
        graph.add_node("finish", lambda state: state)

        # ==================== Edges Setup ====================

        graph.set_entry_point("start")

        # graph.add_edge("start", "suggest_page_fixes")
        # graph.add_edge("suggest_page_fixes", "extract_page_text")
        """ graph.add_edge("start", "load_document")
        graph.add_edge("load_document", "extract_document_pages_data") """

        graph.add_edge("start", "extract_document_text")
        # graph.add_edge("extract_document_pages_data", "extract_document_text")
        # graph.add_conditional_edges("extract_document_text", self._extract_document_text_conditional_edges)
        graph.add_edge("extract_document_text", "review_document_text")
        graph.add_edge("review_document_text", "questions_markdown_formatter")
        graph.add_edge("questions_markdown_formatter", "extract_document_pages_data")
        graph.add_edge("extract_document_pages_data", "extract_document_clinical_case")

        graph.add_edge("extract_document_clinical_case", "finish")
        graph.add_edge("finish", END)

        return graph.compile()

    async def _load_document_step(self, state: DocumentExtractionState) -> DocumentExtractionState:
      print(f"🔍 Loading Document")

      from langchain_docling import DoclingLoader
      import tempfile
      import os

      for i, img in enumerate(state.exam_images):
        # Create a temporary file for the image
        print(f"🔍 Loading Document Image {i + 1}")
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_img_file:
            temp_img_path = temp_img_file.name
            temp_img_file.write(base64.b64decode(img))

        try:
            loader = DoclingLoader(
                file_path=temp_img_path
            )
            docs = loader.load()

            print(f"🔍 Loaded Document")
            contents = [doc.page_content for doc in docs]

            print(f"🔍 Contents")
            state.document_contents.append("\n".join(contents))

        finally:
            # Unlink (delete) the temporary file after use
            if os.path.exists(temp_img_path):
                os.unlink(temp_img_path)

      return state

    async def _extract_document_text_step(self, state: DocumentExtractionState) -> DocumentExtractionState:
      #current_page_index = state.current_page_index
      #print(f"🔍 Extracting Document Text for Document Page {current_page_index}")
      print(f"🔍 Extracting Document Text for Document")

      # Use the extract_document_text.j2 template for the prompt
      messages = [
          SystemMessage(
              content=[
                  {
                      "type": "text",
                      "text": "You are a helpful assistant that extracts the text from the image of the exam page. You only return the response, not confirmation, no greetings, no explanations, no nothing. Just the final result based on user's request."
                  }
              ]
          ),
          HumanMessage(
              content=[
                  {
                      "type": "text",
                      "text": render_template('extract_document_text')
                  },
                  *[
                      {
                          "type": "image",
                          "source_type": "base64",
                          "data": img,
                          "mime_type": "image/jpeg",
                      }
                      for img in state.exam_images
                  ],
              ]
          )
      ]

      """ structured_llm = self.gLlm.with_structured_output(PageQuestions)
      response: PageQuestions = await structured_llm.ainvoke(messages) """
      response = self.gLlm.invoke(messages)
      print(f"🔍 Extracted Document Text for Page")
      print(response)

      await self.handle_quota()

      state.questions_raw_text = response.content

      return state

    async def _review_document_text_step(self, state: DocumentExtractionState) -> DocumentExtractionState:
      #current_page_index = state.current_page_index
      #print(f"🔍 Extracting Document Text for Document Page {current_page_index}")
      print(f"🔍 Extracting Document Text for Document")

      # Use the extract_document_text.j2 template for the prompt
      messages = [
          HumanMessage(
              content=[
                  {
                      "type": "text",
                      "text": render_template('document_extraction_reviewer', {
                        "initial_extraction": state.questions_raw_text
                      })
                  },
                  *[
                      {
                          "type": "image",
                          "source_type": "base64",
                          "data": img,
                          "mime_type": "image/jpeg",
                      }
                      for img in state.exam_images
                  ],
              ]
          )
      ]

      """ structured_llm = self.gLlm.with_structured_output(PageQuestions)
      response: PageQuestions = await structured_llm.ainvoke(messages) """
      response = self.gLlm.invoke(messages)
      print(f"🔍 Extracted Document Text for Page")
      print(response)

      await self.handle_quota()

      state.questions_raw_text = response.content

      return state

    async def _questions_markdown_formatter_step(self, state: DocumentExtractionState) -> DocumentExtractionState:
      print(f"🔍 Formatting Questions Markdown")

      messages = [
          HumanMessage(
              content=[
                  {
                      "type": "text",
                      "text": render_template('questions_markdown_formatter', {
                        "raw_text": state.questions_raw_text
                      })
                  }
              ]
          )
      ]

      response = self.gLlm.invoke(messages)
      print(f"🔍 Formatted Questions Markdown")
      print(response)

      await self.handle_quota()

      state.questions_markdown_text = response.content

      print(f"🔍 Questions Markdown Text")
      print(state.questions_markdown_text)

      structured_llm = self.gLlm.with_structured_output(PageQuestions)
      response: PageQuestions = await structured_llm.ainvoke([
        SystemMessage(
          content=[
            {
              "type": "text",
              "text": "You are an expert parsing agent. Your task is to accurately parse the user's text, which is a markdown representation of an exam, and extract all the questions into the provided 'PageQuestions' schema. Pay close attention to question numbers, the full question text, and all associated options. questions are delimited by a line containing only '---' before and after the question, and the question number is the first number in the question text, and the options are preceded by a letter and a period, the number of options is the number of letter preceeding it."
            }
          ]
        ),
        HumanMessage(
          content=[
            {
              "type": "text",
              "text": state.questions_markdown_text
            }
          ]
        )
      ])

      state.questions = response.questions

      return state

    async def _extract_document_text_conditional_edges(self, state: DocumentExtractionState) -> str:
      if state.current_page_index < len(state.exam_images):
        return "extract_document_text"
      else:
        return "extract_document_clinical_case"

    async def _extract_document_pages_data_step(self, state: DocumentExtractionState) -> DocumentExtractionState:
      print(f"🔍 Extracting Document Pages Data")

      structured_llm = self.gLlm.with_structured_output(PageDataOutput)

      # Use the extract_document_text.j2 template for the prompt
      messages = [
          SystemMessage(
              content=[
                  {
                      "type": "text",
                      "text": "You are a helpful assistant that extracts the pages data from the image of the exam page. You only return the response, not confirmation, no greetings, no explanations, no nothing. Just the final result based on user's request."
                  }
              ]
          ),
          HumanMessage(
              content=[
                  {
                      "type": "text",
                      "text": render_template('extract_document_pages_data', {
                         "questions_count": len(state.questions)
                      })
                  },
                  *[
                      {
                          "type": "image",
                          "source_type": "base64",
                          "data": img,
                          "mime_type": "image/jpeg",
                      }
                      for img in state.exam_images
                  ],
              ]
          )
      ]

      response: PageDataOutput = await structured_llm.ainvoke(messages)
      print(f"🔍 Extracted Document Pages Data")

      await self.handle_quota()

      state.pages_data = response

      return state

    async def _extract_document_clinical_case_step(self, state: DocumentExtractionState) -> DocumentExtractionState:
      print(f"🔍 Extracting Document Clinical Case")

      structured_llm = self.gLlm.with_structured_output(DocumentClinicalCaseOutput)

      # Use the extract_document_text.j2 template for the prompt
      messages = [
          SystemMessage(
              content=[
                  {
                      "type": "text",
                      "text": "You are a helpful assistant that extracts the clinical case from the image of the exam page. You only return the response, not confirmation, no greetings, no explanations, no nothing. Just the final result based on user's request."
                  }
              ]
          ),
          HumanMessage(
              content=[
                  {
                      "type": "text",
                      "text": render_template('extract_document_clinical_case', {
                        "questions_markdown_text": state.questions_markdown_text
                      })
                  },
                  *[
                    {
                      "type": "image",
                      "source_type": "base64",
                      "data": img,
                      "mime_type": "image/jpeg",
                    } for img in state.exam_images
                  ]
              ]
          )
      ]

      response: DocumentClinicalCaseOutput = await structured_llm.ainvoke(messages)
      print(f"🔍 Extracted Document Clinical Case")

      clinical_cases_list: List[ClinicalCaseResponse] = []
      for clinical_case in response.data:
        print(f"Clinical Case: {clinical_case}")
        clinical_cases_list.append(ClinicalCaseResponse(
          question_numbers=list(range(clinical_case.start_question_number, clinical_case.end_question_number + 1)),
          clinical_case=clinical_case.clinical_case,
          type="clinicalCase",
          images=[]
        ))

      state.pages_clinical_cases = clinical_cases_list

      print("Pages Clinical Cases")
      print(state.pages_clinical_cases)

      await self.handle_quota()

      return state

    async def _format_extracted_data(self, state: DocumentExtractionState) -> Dict[str, Any]:
      print(f"🔍 Formatting Extracted Data")

      return state

    async def handle_quota(self):
        self.gLlmCalls += 1
        time_to_wait = 30

        if self.gLlmCalls % self.max_quota == 0:
            print(f"🔍 Quota Reached, Sleeping for {time_to_wait} seconds")
            await asyncio.sleep(time_to_wait)

    async def run(self, pdf_path: str = None, exam_images: list = None) -> DocumentExtractionState:
        """
        Run the workflow with either a PDF path or pre-extracted images.
        
        Args:
            pdf_path: Path to PDF file (optional)
            exam_images: Pre-extracted images (optional)
        
        Returns:
            ExtractionState: Final state with extracted questions
        """
        if exam_images is None:
          if pdf_path is None:
              import os
              # Fallback to hardcoded path for backward compatibility
              pdf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "file.pdf"))
          exam_images = extract_pdf_pages_as_images(pdf_path=pdf_path)

        initial_state = DocumentExtractionState()
        # Only use page 5 (index 4) from exam_images
        initial_state.exam_images = exam_images
        initial_state.pdf_path = pdf_path
        final_state = await self.workflow.ainvoke(initial_state, {
            "recursion_limit": 120
        })

        return DocumentExtractionState(**final_state)
