"""
Resume Scorer — Stujya AI Features
====================================
Scores a resume out of 100 based on how well it matches a target role.
Returns only overall_score and target_role.
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


SCORING_PROMPT = """You are an expert resume reviewer and career coach.

Analyse the provided resume text and score it out of 100 based on how well it qualifies the candidate for the target role: "{target_role}".

Return ONLY a valid JSON object — no markdown fences, no explanations.

Required JSON schema:
{{
  "overall_score": <integer 0-100>,
  "target_role": "{target_role}"
}}

Scoring guidelines:
- 90-100: Exceptional match, ready for the role
- 75-89: Strong candidate with minor gaps
- 60-74: Decent match but needs improvement in key areas
- 40-59: Below average, significant gaps for this role
- 0-39: Poor match, major reskilling needed

Rules:
- Be honest and constructive, not overly generous.
- Base scores on evidence in the resume, not assumptions.
- Return raw JSON only. No markdown. No code fences. No extra text.
"""


def _extract_pdf_text(file_path: str) -> str:
    """Extract text from a PDF file using PyPDF2."""
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        pdf_text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                pdf_text += page_text + "\n"
    return pdf_text


def _clean_json_response(text: str) -> str:
    """Strip markdown code fences if Gemini wraps the JSON."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def score_resume(file_path: str, target_role: str) -> dict:
    """
    Score a resume PDF against a target role.

    Args:
        file_path: Absolute or relative path to the resume PDF.
        target_role: The role to score against (e.g. "Frontend Developer").

    Returns:
        A dict with keys: overall_score (int, 0-100) and target_role (str).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If inputs are invalid or PDF has no text.
        RuntimeError: If Gemini fails or returns invalid JSON.
    """
    target_role = target_role.strip()

    if not target_role:
        raise ValueError("target_role cannot be empty.")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    if not file_path.lower().endswith(".pdf"):
        raise ValueError(f"Only PDF files are supported. Got: {file_path}")

    file_size = os.path.getsize(file_path)
    if file_size > 5 * 1024 * 1024:
        raise ValueError("File too large. Max 5 MB.")

    try:
        pdf_text = _extract_pdf_text(file_path)
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
        prompt = SCORING_PROMPT.format(target_role=target_role)
        prompt_with_resume = f"{prompt}\n\n--- RESUME TEXT ---\n{pdf_text}\n--- END ---"
        response = model.generate_content(prompt_with_resume)

        raw_text = response.text
        cleaned = _clean_json_response(raw_text)

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"Gemini returned invalid JSON: {e}. Raw: {cleaned[:500]}"
            )

        overall_score = max(0, min(100, int(parsed.get("overall_score", 0))))

        return {
            "overall_score": overall_score,
            "target_role": target_role,
        }

    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to score resume: {str(e)}")
