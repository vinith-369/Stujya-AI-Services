"""
Resume Scorer — Stujya AI Features
====================================
Scores a resume out of 100 based on how well it matches a target role.
Uses PyPDF2 for text extraction and Gemini for analysis.
"""

import os
import json

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import PyPDF2
import google.generativeai as genai


# ── Router ─────────────────────────────────────────────────────────────────────
router = APIRouter()


# ── Request model ──────────────────────────────────────────────────────────────
class ScoreRequest(BaseModel):
    """Request body: file path + target role."""
    file_path: str       # absolute or relative path to the resume PDF
    target_role: str     # e.g. "Frontend Developer", "Data Scientist", "ML Engineer"


# ── Response model ─────────────────────────────────────────────────────────────
class ScoreResponse(BaseModel):
    """Detailed resume score breakdown."""
    overall_score: int                    # 0-100
    target_role: str                      # the role scored against
    summary: str                          # 1-2 sentence overall assessment
    strengths: list[str]                  # what the resume does well for this role
    weaknesses: list[str]                 # gaps or missing areas
    suggestions: list[str]               # actionable improvement tips
    section_scores: dict[str, int]        # e.g. {"skills": 85, "experience": 60, ...}


# ── Gemini prompt ──────────────────────────────────────────────────────────────
SCORING_PROMPT = """You are an expert resume reviewer and career coach.

Analyse the provided resume text and score it out of 100 based on how well it qualifies the candidate for the target role: "{target_role}".

Return ONLY a valid JSON object — no markdown fences, no explanations.

Required JSON schema:
{{
  "overall_score": <integer 0-100>,
  "target_role": "{target_role}",
  "summary": "<string> (1-2 sentence overall assessment of how well the resume matches the target role)",
  "strengths": ["<string>", ...] (3-5 specific strengths relevant to the target role),
  "weaknesses": ["<string>", ...] (3-5 specific gaps or weaknesses for the target role),
  "suggestions": ["<string>", ...] (3-5 actionable tips to improve the resume for this role),
  "section_scores": {{
    "skills_match": <integer 0-100> (how well the skills align with the role),
    "experience_relevance": <integer 0-100> (how relevant the work/project experience is),
    "education": <integer 0-100> (how well education supports the role),
    "projects": <integer 0-100> (quality and relevance of projects),
    "presentation": <integer 0-100> (formatting, clarity, and professionalism)
  }}
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
- Tailor all feedback specifically to the target role.
- Return raw JSON only. No markdown. No code fences. No extra text.
"""


# ── Helpers ────────────────────────────────────────────────────────────────────
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


# ── API endpoint ───────────────────────────────────────────────────────────────
@router.post("/score-resume", response_model=ScoreResponse)
async def score_resume(request: ScoreRequest):
    """
    Provide a local file path to a resume PDF and a target role.
    Returns a detailed score (0-100) with breakdown and suggestions.

    Example request body:
        {"file_path": "/Users/you/Desktop/resume.pdf", "target_role": "Frontend Developer"}
    """
    file_path = request.file_path
    target_role = request.target_role.strip()

    # ── Validate inputs ────────────────────────────────────────────────────
    if not target_role:
        raise HTTPException(status_code=400, detail="target_role cannot be empty.")

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

    if not file_path.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail=f"Only PDF files are supported. Got: {file_path}",
        )

    file_size = os.path.getsize(file_path)
    if file_size > 5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Max 5 MB.")

    # ── Extract text from PDF ──────────────────────────────────────────────
    try:
        pdf_text = _extract_pdf_text(file_path)
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
        prompt = SCORING_PROMPT.format(target_role=target_role)
        prompt_with_resume = f"{prompt}\n\n--- RESUME TEXT ---\n{pdf_text}\n--- END ---"
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

        # ── Clamp scores to 0-100 ─────────────────────────────────────────
        parsed["overall_score"] = max(0, min(100, int(parsed.get("overall_score", 0))))
        if parsed.get("section_scores"):
            for key in parsed["section_scores"]:
                parsed["section_scores"][key] = max(0, min(100, int(parsed["section_scores"][key])))

        result = ScoreResponse(**parsed)
        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to score resume: {str(e)}",
        )
