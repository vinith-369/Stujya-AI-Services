"""
Resume Parser — Stujya AI Features
====================================
A plain function that:
  1. Accepts a resume PDF file path
  2. Sends it to Google Gemini for extraction
  3. Returns a dict matching the StudentSignup form fields
"""

import os
import json

import PyPDF2
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is not set. "
        "Create a .env file in ai_features/ with: GEMINI_API_KEY=your_key_here"
    )

genai.configure(api_key=GEMINI_API_KEY)

VALID_UNIVERSITIES = [
    "IIT Bombay",
    "IIT Delhi",
    "BITS Pilani",
    "NIT Trichy",
    "Anna University",
    "Other",
]


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


def parse_resume(file_path: str) -> dict:
    """
    Parse a resume PDF and return structured data matching the Stujya
    student signup form.

    Args:
        file_path: Absolute or relative path to the resume PDF.

    Returns:
        A dict with keys: firstName, lastName, email, university, major,
        graduationYear, skills, bio.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a PDF, is too large, or has no text.
        RuntimeError: If Gemini fails or returns invalid JSON.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    if not file_path.lower().endswith(".pdf"):
        raise ValueError(f"Only PDF files are supported. Got: {file_path}")

    file_size = os.path.getsize(file_path)
    if file_size > 5 * 1024 * 1024:  # 5 MB limit
        raise ValueError("File too large. Max 5 MB.")

    try:
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            pdf_text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    pdf_text += page_text + "\n"

        if not pdf_text.strip():
            raise ValueError(
                "Could not extract any text from the PDF. "
                "The file may be image-based or corrupted."
            )

    except ValueError:
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to read PDF: {str(e)}")

    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        prompt_with_resume = f"{EXTRACTION_PROMPT}\n\n--- RESUME TEXT ---\n{pdf_text}\n--- END ---"
        response = model.generate_content(prompt_with_resume)

        raw_text = response.text
        cleaned = _clean_json_response(raw_text)

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"Gemini returned invalid JSON: {e}. Raw: {cleaned[:500]}"
            )

        normalised = _validate_and_normalise(parsed)
        return normalised

    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to process resume: {str(e)}")
