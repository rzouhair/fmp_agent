import asyncio
import os
from typing import Dict, Any, List
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv

from src.utils.agent import render_template
from src.utils.pdf import extract_pdf_pages_as_images
from .models import ExtractionState, PageClinicalCases, PageQuestions, PageQuestionsNumbers, PageQuestionsText, Question, QuestionOption
from .prompts import DeveloperToolsPrompts

load_dotenv()


class Workflow:
    def __init__(self):

        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
        # self.gLlm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-04-17", temperature=0.5, google_api_key=os.getenv("GOOGLE_API_KEY"))
        self.gLlm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
        self.gLlmCalls = 0
        self.max_quota = 1000

        self.prompts = DeveloperToolsPrompts()
        self.workflow = self._build_workflow()

    def _build_workflow(self):
        graph = StateGraph(ExtractionState)

        # ==================== Nodes Setup ====================

        graph.add_node("start", lambda state: state)
        # graph.add_node("suggest_page_fixes", self._suggest_page_fixes_step)
        # graph.add_node("review_page_text", self._review_page_text_step)
        graph.add_node("extract_page_text", self._extract_questions_text_step)
        graph.add_node("extract_questions_numbers", self._extract_questions_numbers_step)
        graph.add_node("review_questions_numbers", self._review_questions_numbers_step)
        graph.add_node("advance_page", self._advance_page_step)
        graph.add_node("extract_clinical_cases", self._extract_clinical_cases_step)
        graph.add_node("remove_clinical_cases_from_page_text", self._remove_clinical_cases_from_page_text_step)
        graph.add_node("concatenate_pages_text", self._concatenate_pages_text_step)
        graph.add_node("extract_questions_using_numbers", self._extract_questions_using_numbers_step)
        graph.add_node("extract_each_page_questions", self._extract_each_page_questions_step)

        # ==================== Edges Setup ====================

        graph.set_entry_point("start")

        # graph.add_edge("start", "suggest_page_fixes")
        # graph.add_edge("suggest_page_fixes", "extract_page_text")
        graph.add_edge("start", "extract_page_text")
        graph.add_edge("extract_page_text", "review_page_text")
        graph.add_edge("review_page_text", "extract_questions_numbers")
        graph.add_edge("extract_questions_numbers", "review_questions_numbers")
        graph.add_edge("review_questions_numbers", "extract_clinical_cases")

        graph.add_conditional_edges(
            "extract_clinical_cases",
            lambda state: (
                "remove_clinical_cases_from_page_text"
                if len(state.pages_clinical_cases_map.get(f"{state.current_page_index}", [])) > 0 and state.current_page_index < len(state.exam_images) - 1
                else (
                    "advance_page"
                    if state.current_page_index < len(state.exam_images) - 1 and len(state.pages_clinical_cases_map.get(f"{state.current_page_index}", [])) <= 0
                    else END
                )
            ),
            {
                "remove_clinical_cases_from_page_text": "remove_clinical_cases_from_page_text",
                "advance_page": "advance_page",
                END: "concatenate_pages_text"
            }
        )


        graph.add_conditional_edges(
          "remove_clinical_cases_from_page_text",
          lambda state: state.current_page_index < len(state.exam_images) - 1,
          {
            True: "advance_page",
            False: "concatenate_pages_text"
          }
        )

        graph.add_edge("concatenate_pages_text", "extract_questions_using_numbers")
        graph.add_edge("extract_questions_using_numbers", "extract_each_page_questions")

        graph.add_conditional_edges(
          "extract_each_page_questions",
          lambda state: state.current_page_index < len(state.exam_images) - 1,
          {
            True: "advance_page",
            False: END
          }
        )

        graph.add_conditional_edges(
          "advance_page",
          lambda state: state.current_phase,
          {
            "extract_page_text": "extract_page_text",
            "extract_each_page_questions": "extract_questions_using_numbers"
          }
        )

        return graph.compile()


    async def _suggest_page_fixes_step(self, state: ExtractionState) -> Dict[str, Any]:
        current_page_index = state.current_page_index
        current_page_image = state.exam_images[current_page_index]
        print(f"🔍 Suggesting Page {current_page_index + 1} Fixes")

        messages = [
          SystemMessage(
              content=[
                  { "type": "text", "text": "You are a helpful assistant that suggests page fixes for the image of the exam page. You only return the response, not confirmation, no greetings, no explanations, no nothing. Just the final result based on user's request." }
              ]
          ),
          HumanMessage(
              content=[
                  {"type": "text", "text": render_template('suggest_page_fixes')},
                  {
                      "type": "image",
                      "source_type": "base64",
                      "quality": "high",
                      "data": current_page_image,
                      "mime_type": "image/jpeg",
                  },
              ],
          )
        ]

        response = await self.llm.ainvoke(messages)

        print(f"🔍 Suggested Page {current_page_index + 1} Fixes: {response}")

        state.pages_fixes_map[f"{current_page_index}"] = response.content

        return state

    async def _review_page_text_step(self, state: ExtractionState) -> Dict[str, Any]:
        current_page_index = state.current_page_index
        current_page_image = state.exam_images[current_page_index]
        current_page_text = state.pages_text[current_page_index]
        print(f"🔍 Reviewing Page {current_page_index + 1} Text")

        messages = [
          SystemMessage(
              content=[
                  { "type": "text", "text": "You are a helpful assistant that reviews the text extracted from the image of the exam page. You only return the response, not confirmation, no greetings, no explanations, no nothing. Just the final result based on user's request." }
              ]
          ),
          AIMessage(
              content=[
                  {"type": "text", "text": "I will never remove any text, like 'ce patient, le patient, etc.', it is totally important to keep it and totally forbidden to remove it, and will never consider instructions, or text with no options as MCQ questions, I guarantee you that I will never remove any text, and I will never consider instructions, or text with no options as MCQ questions"}
              ]
          ),
          HumanMessage(
              content=[
                  {"type": "text", "text": render_template('review_page_text', {
                    "extracted_text": current_page_text
                  })},
                  {
                      "type": "image",
                      "source_type": "base64",
                      "quality": "high",
                      "data": current_page_image,
                      "mime_type": "image/jpeg",
                  },
              ],
          )
        ]

        response = await self.gLlm.ainvoke(messages)
        print(f"🔍 Reviewed Page {current_page_index + 1}")

        await self.handle_quota()

        state.pages_text[current_page_index] = response.content

        return state

    async def _extract_questions_text_step(self, state: ExtractionState) -> Dict[str, Any]:
        current_page_index = state.current_page_index
        current_page_image = state.exam_images[current_page_index]
        if current_page_index > 0:
            total_prev_questions = sum(
                len(state.pages_questions_map[str(i)].question_numbers)
                for i in range(current_page_index)
                if str(i) in state.pages_questions_map
            ) + 1
        else:
            total_prev_questions = 1

        print(f"🔍 Previous Page Questions Numbers: {total_prev_questions}")
        # current_page_fixes = state.pages_fixes_map[f"{current_page_index}"]
        print(f"🔍 Extracting Page {current_page_index + 1} Texts Content")

        messages = [
          SystemMessage(
              content=[
                  { "type": "text", "text": "You are a helpful assistant that extracts the text from the image of the exam page. You only return the response, not confirmation, no greetings, no explanations, no nothing. Just the final result based on user's request." }
              ]
          ),
          AIMessage(
              content=[
                  {"type": "text", "text": "I will never remove any text, like 'ce patient, le patient, etc.', it is totally important to keep it and totally forbidden to remove it, and will never consider instructions, or text with no options as MCQ questions, I guarantee you that I will never remove any text, and I will never consider instructions, or text with no options as MCQ questions"}
              ]
          ),
          HumanMessage(
              content=[
                  {"type": "text", "text": render_template('extract_page_text', {
                    "start_number": total_prev_questions
                  })},
                  {
                      "type": "image",
                      "source_type": "base64",
                      "data": current_page_image,
                      "mime_type": "image/jpeg",
                  },
              ],
          )
        ]

        response = await self.gLlm.ainvoke(messages)
        print(f"🔍 Extracted Page {current_page_index + 1}")
        # Print token usage details if available in the response
        usage = response.response_metadata.get('usage_metadata', None)
        if usage:
            input_tokens = usage.get("input_tokens", "N/A")
            output_tokens = usage.get("output_tokens", "N/A")
            total_tokens = usage.get("total_tokens", "N/A")
            print(f"🔢 LLM Token Usage: input={input_tokens}, output={output_tokens}, total={total_tokens}")
        else:
            print(f"🔢 LLM Token Usage: (usage_metadata not available) {response}")

        await self.handle_quota()

        state.pages_text.append(response.content)

        return state

    async def _extract_questions_numbers_step(self, state: ExtractionState) -> Dict[str, Any]:
        current_page_index = state.current_page_index
        current_page_text = state.pages_text[current_page_index]
        print(f"🔍 Extracting Questions Numbers in Page {current_page_index + 1}")

        structured_llm = self.llm.with_structured_output(PageQuestionsNumbers)
        
        messages = [
            SystemMessage(
                content=[
                    { "type": "text", "text": "You are a helpful assistant that extracts the questions numbers from the text of the exam page. You only return the response, not confirmation, no greetings, no explanations, no nothing. Just the final result based on user's request." }
                ]
            ),
            AIMessage(
                content=[
                    {"type": "text", "text": """
I will be strict with the question numbers extraction, I understand that every question should follow a format like this in order for its number to be extracted:

1- L'examen clinique est normal à part la paralysie faciale périphérique. Vous retenez le diagnostic de la Paralysie faciale à frigo. Le traitement de première intention peut faire appel à :
A- La corticothérapie par voie générale
B- Les anti inflammatoire non stéroïdiens
C- L'antibiothérapie probabiliste
D- La décompression chirurgicale
E- La kinésithérapie faciale

I will never consider instructions, questions with no options, questions preceeded with numberings, or options preceeded by an alphabetical prefix as MCQ questions, I will follow the rules strictly, I will follow also the format like the example below to extract MCQ questions, any format other than these, I will exclude them:
"""}
                ]
            ),
            HumanMessage(
                content=[
                    {
                      "type": "text",
                      "text": render_template('extract_page_numbers', {
                        "input_text": current_page_text
                      })
                  },
                ]
            )
        ]

        response: PageQuestionsNumbers = await structured_llm.ainvoke(messages)

        state.pages_questions_map[f"{current_page_index}"] = response if len(response.question_numbers) <= 8 else PageQuestionsNumbers(question_numbers=[])

        print(f"🔍 Extracted Questions Numbers: {response}")

        return state

    async def _review_questions_numbers_step(self, state: ExtractionState) -> Dict[str, Any]:
        current_page_index = state.current_page_index
        current_page_questions_numbers = state.pages_questions_map[f"{current_page_index}"]
        print(f"🔍 Reviewing Questions Numbers: {current_page_questions_numbers}")

        structured_llm = self.llm.with_structured_output(PageQuestionsNumbers)
        
        messages = [
          SystemMessage(
              content=[
                  { "type": "text", "text": "You are a helpful assistant that reviews the questions numbers extracted from the text of the exam page. You only return the response, not confirmation, no greetings, no explanations, no nothing. Just the final result based on user's request." }
              ]
          ),
          HumanMessage(
              content=[
                  { "type": "text", "text": render_template('review_questions_numbers', {
                    "input_text": current_page_questions_numbers,
                    "extracted_result": state.pages_questions_map[f"{current_page_index}"]
                  }) }
              ]
          )
        ]

        response: PageQuestionsNumbers = await structured_llm.ainvoke(messages)

        state.pages_questions_map[f"{current_page_index}"] = response if len(response.question_numbers) <= 8 else PageQuestionsNumbers

        print(f"🔍 Reviewed Questions Numbers: {response}")
        
        return state

    async def _advance_page_step(self, state: ExtractionState) -> Dict[str, Any]:
        if state.current_page_index < len(state.exam_images) - 1:
          state.current_page_index += 1

        else:
          state.current_phase = "extract_each_page_questions"

        return state

    async def _extract_clinical_cases_step(self, state: ExtractionState) -> Dict[str, Any]:
        current_page_index = state.current_page_index
        current_page_text = state.pages_text[current_page_index]
        
        structured_llm = self.gLlm.with_structured_output(PageClinicalCases)

        messages = [
          SystemMessage(
              content=[
                  { "type": "text", "text": "You are a helpful assistant that extracts the clinical cases from the text of the exam page. You only return the response, not confirmation, no greetings, no explanations, no nothing. Just the final result based on user's request." }
              ]
          ),
          AIMessage(
              content=[
                  {"type": "text", "text": """
I will never remove any text, like 'ce patient, le patient, etc.', I will be as strict as possible, I will never consider instructions, questions with no options, questions preceeded with numberings, or options preceeded by an alphabetical prefix as clinical cases, I will follow the rules strictly, I will follow the explanation below to extract clinical cases, any format other than these, I will exclude them:
                   
Clinical cases are generally narrative paragraphs describing a patient's situation (age, sex, medical history, symptoms, clinical context, examination results, etc.), followed by one or more questions about management, diagnosis, or treatment. They do not include question numbering or answer options, and are characterized by their descriptive structure focused on a specific patient or clinical situation.
"""}
              ]
          ),
          HumanMessage(
              content=[
                  { "type": "text", "text": render_template('extract_clinical_cases', {
                    "page_text_content": current_page_text
                  }) }
              ]
          )
        ]

        response = await structured_llm.ainvoke(messages)

        state.pages_clinical_cases_map[f"{current_page_index}"] = response.clinical_cases

        print(f"🔍 Extracted Clinical Cases: {response}")

        await self.handle_quota()

        return state

    async def _remove_clinical_cases_from_page_text_step(self, state: ExtractionState) -> Dict[str, Any]:
        print(f"Skipping Clinical Cases Removal")
        current_page_index = state.current_page_index
        current_page_text = state.pages_text[current_page_index]
        current_page_clinical_cases: List[str] = state.pages_clinical_cases_map[f"{current_page_index}"]

        state.pages_text[current_page_index] = current_page_text
        
        messages = [
          SystemMessage(
              content=[
                  { "type": "text", "text": "You are a helpful assistant that removes the clinical cases from the text of the exam page. You only return the response, not confirmation, no greetings, no explanations, no nothing. Just the final result based on user's request." }
              ]
          ),
          HumanMessage(
              content=[
                  { "type": "text", "text": render_template('remove_clinical_cases_from_page_text', {
                    "page_text_content": current_page_text,
                    "clinical_cases": "\n===========\n".join(map(lambda clinical_case: clinical_case, current_page_clinical_cases))
                  }) }
              ]
          )
        ]

        response = await self.llm.ainvoke(messages)

        state.pages_text[current_page_index] = response.content

        print(f"🔍 Removed Clinical Cases from Page {current_page_index + 1}")

        return state

    async def _concatenate_pages_text_step(self, state: ExtractionState) -> Dict[str, Any]:
        state.current_page_index = 0
        state.pages_text_concatenated = "\n".join(state.pages_text)

        print("--------------------------------")
        print(f"🔍 Concatenated Pages Text")
        print("--------------------------------")

        state.current_phase = "extract_each_page_questions"

        return state

    async def _extract_questions_using_numbers_step(self, state: ExtractionState) -> Dict[str, Any]:
        current_page_index = state.current_page_index
        if current_page_index == 0:
            if len(state.pages_text) > 1:
                concatenated_pages_text = state.pages_text[0] + "\n" + state.pages_text[1]
            else:
                concatenated_pages_text = state.pages_text[0]
        elif current_page_index == len(state.pages_text) - 1:
            concatenated_pages_text = state.pages_text[current_page_index - 1] + "\n" + state.pages_text[current_page_index]
        else:
            concatenated_pages_text = state.pages_text[current_page_index] + "\n" + state.pages_text[current_page_index + 1]

        print("--------------------------------")
        print(f"🔍 Concatenated Pages Text: {concatenated_pages_text}")
        print("--------------------------------")
        print(f"Concatenated Pages Text: {concatenated_pages_text}")
        print("--------------------------------")

        current_page_questions_numbers = state.pages_questions_map[f"{current_page_index}"]
        print(f"🔍 Extracting Questions Using Numbers: {current_page_questions_numbers}")

        structured_llm = self.llm.with_structured_output(PageQuestionsText)

        messages = [
          SystemMessage(
              content=[
                  { "type": "text", "text": "You are a helpful assistant that extracts the questions using the numbers from the text of the exam page. You only return the response, not confirmation, no greetings, no explanations, no nothing. Just the final result based on user's request." }
              ]
          ),
          HumanMessage(
              content=[
                  { "type": "text", "text": render_template('extract_questions_using_numbers', {
                    "question_numbers": current_page_questions_numbers,
                    "page_text_content": concatenated_pages_text.replace("`", "")
                  }) }
              ]
          )
        ]

        response: PageQuestionsText = await structured_llm.ainvoke(messages)

        state.pages_questions_text_map[f"{current_page_index}"] = response.questions

        # print(f"🔍 Extracted Questions and Clinical Cases: {response}")

        return state
      

    async def _extract_each_page_questions_step(self, state: ExtractionState) -> Dict[str, Any]:
        current_page_index = state.current_page_index
        all_questions_pages_text = state.pages_questions_text_map[f"{current_page_index}"]
        current_page_map = state.pages_questions_map[f"{current_page_index}"]
        print(f"🔍 Extracting Each Page Questions: {current_page_index}")
        print(all_questions_pages_text)
        print(f"Where is the question numbers are? {current_page_map.question_numbers}")

        structured_llm = self.gLlm.with_structured_output(PageQuestions)

        hasFailed = False
        missingQuestions = []
        retryCount = 3

        extracted_question_numbers = []
        while retryCount > 0:

          messages = [
            SystemMessage(
                content=[
                    { "type": "text", "text": "You are a helpful assistant that extracts the questions and their numbering from the text content of the exam page. You only return the response, not confirmation, no greetings, no explanations, no nothing. Just the final result based on user's request." }
                ]
            ),
            AIMessage(
                content=[
                    {"type": "text", "text": f"I will follow the instructions strictly, and I will extract the exact questions apprating in the questions numbering list, not more, not less, I will extract exactly {len(current_page_map.question_numbers)} questions with the numberings {current_page_map.question_numbers}, and I will review it thouroughly before returning the result, I repeat, exactly {len(current_page_map.question_numbers)} my output should definitely contain the questions with the numberings [{current_page_map.question_numbers}], otherwise I will be punished!!"}
                ]
            ),
            HumanMessage(
                content=[
                    { "type": "text", "text": render_template('extract_page_questions', {
                      "page_text_content": all_questions_pages_text,
                      "questions_numbering": current_page_map.question_numbers,
                      "review": f"Why are you so stupid? You have returned the wrong number of questions, please try again, you returned {len(response.questions)} questions, but I asked you to return {len(current_page_map.question_numbers)} questions which are {current_page_map.question_numbers}, the missing questions are {missingQuestions}, if you don't include them, You will be punished, and fired!!!!" if hasFailed else ""
                    }) }
                ]
            )
          ]

          response: PageQuestions = await structured_llm.ainvoke(messages)
          extracted_question_numbers = [question.number for question in response.questions]

          print(f"🔍 Extracted Questions: {extracted_question_numbers}")

          await self.handle_quota()

          if len(extracted_question_numbers) == len(current_page_map.question_numbers):
            hasFailed = False
            missingQuestions = []
            break
          else:
            print(f"🔍 Failed to extract questions, retrying... {retryCount} retries left")
            missingQuestions = [question for question in current_page_map.question_numbers if question not in extracted_question_numbers]
            print(f"Missing questions: {missingQuestions}")
            hasFailed = True
            retryCount -= 1


        extracted_question_numbers = [question.number for question in response.questions]
        print("--------------------------------")
        print(f"🔍 Existing Questions: {len(state.exam_questions)}")
        print(f"🔍 New Questions: {len(response.questions)}")
        print(response.questions)
        print(f"🔍 Extracted question numbers: {extracted_question_numbers}")

        filling_questions = []
        questions_count_difference = abs(len(current_page_map.question_numbers) - len(response.questions))
        missing_sequential_questions = []

        # Add difference between the last question in the current_page_map.question_numbers and the next page's first question number to questions_count_difference
        if current_page_map.question_numbers:
            last_current_question = current_page_map.question_numbers[-1]
            next_first_question = None
            if current_page_index + 1 < len(state.exam_images):
                next_page_map = state.pages_questions_map.get(f"{current_page_index + 1}")
                if next_page_map and hasattr(next_page_map, "question_numbers") and next_page_map.question_numbers:
                    next_first_question = next_page_map.question_numbers[0]
            if next_first_question is not None:
                diff = next_first_question - last_current_question
                # Add the number of missing questions between the last question of the current page and the first question of the next page.
                # For example, if last_current_question is 10 and next_first_question is 13, then questions 11 and 12 are missing (diff - 1 = 2).
                if diff > 1 and diff < 8:
                    questions_count_difference += (diff - 1)

        print(f"🔍 Missing sequential questions across current and next page: {missing_sequential_questions}")

        if 0 < questions_count_difference < 8:
          filling_options: List[QuestionOption] = [QuestionOption(option="Filling Option") for _ in range(5)]
          filling_questions: List[Question] = [Question(question="Filling Question", options=filling_options, number=0) for _ in range(questions_count_difference)]

          
        elif 0 < len(missing_sequential_questions) < 8:
          filling_options: List[QuestionOption] = [QuestionOption(option="Filling Option") for _ in range(5)]
          filling_questions: List[Question] = [Question(question="Filling Question", options=filling_options, number=0) for _ in range(len(missing_sequential_questions))]

        new_questions = response.questions
        merged_questions = state.exam_questions + new_questions + filling_questions

        state.exam_questions = merged_questions

        filling_questions = []
        print(f"❌ Filling Questions: {len(filling_questions)}") if len(filling_questions) > 0 else print("✅ No Filling Questions")
        print(f"🔍 Total Questions: {len(merged_questions)}")
        print("--------------------------------")

        if current_page_index >= len(state.exam_images) - 1:
          print(state.exam_questions)
        
        return state

    async def handle_quota(self):
        self.gLlmCalls += 1
        time_to_wait = 30

        if self.gLlmCalls % self.max_quota == 0:
            print(f"🔍 Quota Reached, Sleeping for {time_to_wait} seconds")
            await asyncio.sleep(time_to_wait)

    async def run(self, pdf_path: str = None, exam_images: list = None) -> ExtractionState:
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

        initial_state = ExtractionState()
        # Only use page 5 (index 4) from exam_images
        initial_state.exam_images = exam_images
        final_state = await self.workflow.ainvoke(initial_state, {
            "recursion_limit": 120
        })
        return ExtractionState(**final_state)
