# PDF Question Extractor API

A FastAPI-based service that extracts multiple choice questions from PDF exam files using AI-powered OCR and text processing.

## Features

- 📄 **PDF Upload**: Accept PDF files via HTTP endpoint
- 🔍 **AI-Powered Extraction**: Uses OpenAI GPT-4 to extract questions and options
- 📋 **Structured Output**: Returns questions in structured JSON format
- 🔧 **Template System**: Uses Jinja2 templates for prompt engineering
- 🚀 **Fast Processing**: Asynchronous processing with FastAPI

## Setup

### Prerequisites

- Python 3.8+
- OpenAI API key
- poppler-utils (for PDF processing)

### Installation

1. Clone the repository and navigate to the project directory

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Install system dependencies (for PDF processing):

**Ubuntu/Debian:**
```bash
sudo apt-get install poppler-utils
```

**macOS:**
```bash
brew install poppler
```

**Windows:**
Download and install poppler from: https://poppler.freedesktop.org/

4. Set up environment variables:
```bash
export OPENAI_API_KEY="your-openai-api-key-here"
```

Or create a `.env` file:
```
OPENAI_API_KEY=your-openai-api-key-here
```

## Usage

### Starting the API Server

```bash
python run_api.py
```

The API will be available at:
- **API Root**: http://localhost:8000
- **Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

### API Endpoints

#### POST /extract-questions

Extract questions from a PDF file.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: PDF file upload

**Response:**
```json
{
  "success": true,
  "total_questions": 10,
  "questions": [
    {
      "question": "What is the primary function of the heart?",
      "options": [
        "To filter blood",
        "To pump blood throughout the body", 
        "To produce red blood cells",
        "To store oxygen"
      ]
    }
  ],
  "message": "Successfully extracted 10 questions from exam.pdf"
}
```

### Example Usage with curl

```bash
curl -X POST "http://localhost:8000/extract-questions" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@/path/to/your/exam.pdf"
```

### Example Usage with Python requests

```python
import requests

url = "http://localhost:8000/extract-questions"
files = {"file": ("exam.pdf", open("exam.pdf", "rb"), "application/pdf")}

response = requests.post(url, files=files)
result = response.json()

print(f"Extracted {result['total_questions']} questions")
for i, question in enumerate(result['questions'], 1):
    print(f"\nQuestion {i}: {question['question']}")
    for j, option in enumerate(question['options'], 1):
        print(f"  {chr(64+j)}. {option}")
```

## Architecture

### Components

- **`src/api.py`**: FastAPI application and endpoints
- **`src/workflow.py`**: Main processing workflow using LangGraph
- **`src/utils/agent.py`**: Template rendering utilities
- **`src/utils/pdf.py`**: PDF processing and image extraction
- **`src/models.py`**: Pydantic models for data validation
- **`src/templates/`**: Jinja2 templates for AI prompts

### Processing Flow

1. **PDF Upload**: Client uploads PDF file
2. **PDF Processing**: Convert PDF pages to base64-encoded images
3. **AI Processing**: Use GPT-4 Vision to extract text and questions
4. **Structured Extraction**: Parse questions and options into structured format
5. **JSON Response**: Return formatted questions as JSON

## Templates

The system uses Jinja2 templates in `src/templates/` for prompt engineering:

- `extract_page_text.j2`: Extract text from PDF page images
- `extract_page_numbers.j2`: Identify question numbers
- `extract_clinical_cases.j2`: Extract clinical case contexts
- And more...

## Development

### Running in Development Mode

```bash
python run_api.py
```

This starts the server with auto-reload enabled.

### API Documentation

Visit http://localhost:8000/docs for interactive API documentation powered by Swagger UI.

## Error Handling

The API includes comprehensive error handling:

- **400**: Invalid file format (non-PDF files)
- **500**: Processing errors (OCR failures, AI API issues, etc.)

## Performance Notes

- Processing time depends on PDF size and complexity
- Typical processing: 2-5 seconds per page
- Large PDFs may take several minutes
- Consider implementing async queuing for production use

## License

This project is for educational and research purposes.
