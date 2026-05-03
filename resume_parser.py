"""
Resume Parser API — Stujya AI Features
=======================================
A single-file FastAPI service that:
  1. Accepts a resume PDF upload
  2. Sends it to Google Gemini for extraction
  3. Returns a JSON response matching the StudentSignup form fields
"""

import os
import json
from typing import Optional

from fastapi import FastAPI, HTTPException
import PyPDF2
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from dotenv import load_dotenv

# ── Load environment variables ─────────────────────────────────────────────────
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is not set. "
        "Create a .env file in ai_features/ with: GEMINI_API_KEY=your_key_here"
    )

genai.configure(api_key=GEMINI_API_KEY)

# ── App setup ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Stujya Resume Parser",
    description="Extracts student signup fields from a resume PDF using Gemini AI",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Include routers from other files ───────────────────────────────────────────
try:
    from ai_features.resume_scorer import router as scorer_router
except ImportError:
    from resume_scorer import router as scorer_router
app.include_router(scorer_router)

# ── Valid values (mirrored from StudentSignup.tsx) ─────────────────────────────
VALID_UNIVERSITIES = [
    "IIT Bombay",
    "IIT Delhi",
    "BITS Pilani",
    "NIT Trichy",
    "Anna University",
    "Other",
]



# ── Response model ─────────────────────────────────────────────────────────────
class ResumeData(BaseModel):
    """
    Matches the formData state in StudentSignup.tsx exactly.
    Fields that cannot be extracted from a resume are left as defaults.
    """
    firstName: str = ""
    lastName: str = ""
    email: str = ""
    university: str = ""         # one of VALID_UNIVERSITIES or ""
    major: str = ""              # e.g. "Computer Science"
    graduationYear: str = ""     # e.g. "2026"
    skills: list[str] = []       # subset of VALID_SKILLS
    bio: str = ""                # short auto-generated bio


# ── Gemini prompt ──────────────────────────────────────────────────────────────
EXTRACTION_PROMPT = """You are an expert resume parser for a student platform called Stujya.

Analyse the attached resume PDF and extract the following fields.
Return ONLY a valid JSON object — no markdown fences, no explanations.

Required JSON schema:
{{
  "firstName": "<string>",
  "lastName": "<string>",
  "email": "<string> (any email found in the resume)",
  "university": "<string> (MUST be one of: {universities}, or empty string if not found / not in the list)",
  "major": "<string> (the student's major / field of study, e.g. Computer Science)",
  "graduationYear": "<string> (four-digit year, e.g. '2026', or empty string if not found)",
  "skills": [<list of strings> (Extract ALL technical and soft skills mentioned in the resume. Include programming languages, frameworks, tools, platforms, methodologies, and soft skills.)],
  "bio": "<string> (generate a short 1-2 sentence professional bio summarising the student based on their resume)"
}}

Rules:
- If a field is not found in the resume, return an empty string (or empty list for skills).
- For university, try to match the college name to one of the valid options. If no match, use "Other".
- For skills, extract ALL skills found in the resume. Be thorough — include programming languages, frameworks, libraries, tools, databases, cloud platforms, soft skills, etc.
- For graduationYear, use the expected graduation year. If only enrollment year is found, estimate graduation (enrollment + 4).
- Return raw JSON only. No markdown. No code fences. No extra text.
""".format(
    universities=", ".join(VALID_UNIVERSITIES),
)


# ── Helpers ────────────────────────────────────────────────────────────────────
def _clean_json_response(text: str) -> str:
    """Strip markdown code fences if Gemini wraps the JSON."""
    text = text.strip()
    if text.startswith("```"):
        # Remove opening fence (```json or ```)
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def _validate_and_normalise(data: dict) -> dict:
    """Ensure values match what the frontend expects."""
    # Validate university
    if data.get("university") and data["university"] not in VALID_UNIVERSITIES:
        data["university"] = "Other"

    # Deduplicate skills while preserving order
    if data.get("skills"):
        seen = set()
        deduped = []
        for skill in data["skills"]:
            if skill.lower() not in seen:
                seen.add(skill.lower())
                deduped.append(skill)
        data["skills"] = deduped

    # Ensure graduationYear is a string
    if data.get("graduationYear"):
        data["graduationYear"] = str(data["graduationYear"])

    return data


# ── Request model ──────────────────────────────────────────────────────────────
class ResumeRequest(BaseModel):
    """Request body: just the file path to the resume PDF."""
    file_path: str  # absolute or relative path to the resume PDF


# ── API endpoint ───────────────────────────────────────────────────────────────
@app.post("/parse-resume", response_model=ResumeData)
async def parse_resume(request: ResumeRequest):
    """
    Provide a local file path to a resume PDF and get back structured JSON
    matching the Stujya student signup form.

    Example request body:
        {"file_path": "/Users/you/Desktop/resume.pdf"}
    """
    file_path = request.file_path

    # ── Validate file exists ───────────────────────────────────────────────
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

    if not file_path.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail=f"Only PDF files are supported. Got: {file_path}",
        )

    file_size = os.path.getsize(file_path)
    if file_size > 5 * 1024 * 1024:  # 5 MB limit
        raise HTTPException(status_code=400, detail="File too large. Max 5 MB.")

    # ── Extract text from PDF ──────────────────────────────────────────────
    try:
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            pdf_text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    pdf_text += page_text + "\n"

        if not pdf_text.strip():
            raise HTTPException(
                status_code=400,
                detail="Could not extract any text from the PDF. The file may be image-based or corrupted.",
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to read PDF: {str(e)}",
        )

    # ── Send text to Gemini ────────────────────────────────────────────────
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        prompt_with_resume = f"{EXTRACTION_PROMPT}\n\n--- RESUME TEXT ---\n{pdf_text}\n--- END ---"
        response = model.generate_content(prompt_with_resume)

        # ── Parse response ─────────────────────────────────────────────────
        raw_text = response.text
        cleaned = _clean_json_response(raw_text)

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=502,
                detail=f"Gemini returned invalid JSON: {e}. Raw: {cleaned[:500]}",
            )

        # ── Validate & normalise ───────────────────────────────────────────
        normalised = _validate_and_normalise(parsed)
        result = ResumeData(**normalised)
        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process resume: {str(e)}",
        )


# ── Health check ───────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "service": "stujya-resume-parser"}


# ── Run directly ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
