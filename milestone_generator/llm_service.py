"""
llm_service.py
--------------
LangGraph-powered milestone generation pipeline.
Uses Groq API (llama-3.3-70b-versatile) — free tier: 14,400 req/day.

Graph flow:
    START → plan_milestones_node → END

Single node: one LLM call that reads real team member details
and generates personalised milestones per member.
"""

import os
import json
from typing import TypedDict, List, Any
from dotenv import load_dotenv

load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END

# ---------------------------------------------------------------------------
# LLM setup
# ---------------------------------------------------------------------------

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7,
    api_key=os.environ.get("GROQ_API_KEY"),
)


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------

class MilestoneState(TypedDict):
    # Inputs
    project_title: str
    short_description: str
    detailed_description: str
    category: str
    difficulty_level: str
    team_members: List[Any]        # flexible — any shape the caller provides
    required_skills: List[str]
    project_budget: str
    estimated_duration: str
    application_deadline: str
    # Output
    milestone_plans: List[dict]


# ---------------------------------------------------------------------------
# Helper: strip markdown fences and parse JSON
# ---------------------------------------------------------------------------

def _parse_json(raw: str) -> dict | list:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(l for l in lines if not l.strip().startswith("```")).strip()
    return json.loads(text)


def _format_team_members(members: List[Any]) -> str:
    """
    Convert the team_members list to a readable string for the prompt.
    Works regardless of member shape (dict, string, nested object, etc.).
    """
    lines = []
    for i, member in enumerate(members, start=1):
        if isinstance(member, dict):
            # Pretty-print key-value pairs for dict members
            details = ", ".join(f"{k}: {v}" for k, v in member.items())
            lines.append(f"  Member {i}: {details}")
        else:
            # Fallback for plain strings or anything else
            lines.append(f"  Member {i}: {member}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Single node: generate milestones for each real team member
# ---------------------------------------------------------------------------

def plan_milestones(state: MilestoneState) -> MilestoneState:
    skills_str = ", ".join(state["required_skills"]) if state["required_skills"] else "N/A"
    team_str   = _format_team_members(state["team_members"])

    prompt = f"""
You are an expert project manager planning a student software project.

PROJECT DETAILS:
- Title: {state["project_title"]}
- Short Description: {state["short_description"]}
- Detailed Description: {state["detailed_description"]}
- Category: {state["category"]}
- Difficulty: {state["difficulty_level"]}
- Required Skills: {skills_str}
- Budget: {state.get("project_budget") or "N/A"}
- Estimated Duration: {state.get("estimated_duration") or "N/A"}
- Application Deadline: {state.get("application_deadline") or "N/A"}

TEAM MEMBERS (with their details / specialisations):
{team_str}

TASK:
For EACH team member listed above, generate a personalised milestone plan with 4–6 milestones that:
  - Leverage that member's specific skills / specialisation.
  - Progress logically: research → design → build → test → deploy/review.
  - Fit within the project timeline.
  - Do NOT give the same milestones to every member — tailor them to what each person is best at.

Each milestone must have:
  - milestone_number  (integer, starts at 1 per member)
  - title             (short, action-oriented)
  - description       (2–3 sentences specific to the member's role/skills)
  - due_date          (relative, e.g. "Week 2", "End of Month 1")
  - assigned_to       (the member's name or identifier from the input)

Return ONLY valid JSON — no markdown, no extra text.

FORMAT:
{{
  "milestone_plans": [
    {{
      "member": "<name or identifier from input>",
      "specialisation": "<their role / skill set>",
      "milestones": [
        {{
          "milestone_number": 1,
          "title": "<title>",
          "description": "<description>",
          "due_date": "<due_date>",
          "assigned_to": "<name or identifier>"
        }}
      ]
    }}
  ]
}}
""".strip()

    response = llm.invoke([HumanMessage(content=prompt)])
    parsed   = _parse_json(response.content)

    state["milestone_plans"] = parsed["milestone_plans"]
    return state


# ---------------------------------------------------------------------------
# Build the LangGraph (single node)
# ---------------------------------------------------------------------------

def _build_graph() -> StateGraph:
    graph = StateGraph(MilestoneState)
    graph.add_node("plan_milestones", plan_milestones)
    graph.add_edge(START, "plan_milestones")
    graph.add_edge("plan_milestones", END)
    return graph.compile()


_graph = _build_graph()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_milestone_plans(project_data: dict) -> dict:
    """
    Accepts project details including a flexible `team_members` list.
    Makes exactly ONE LLM call to generate per-member milestone plans.
    """
    initial_state: MilestoneState = {
        "project_title":        project_data.get("project_title", ""),
        "short_description":    project_data.get("short_description", ""),
        "detailed_description": project_data.get("detailed_description", ""),
        "category":             project_data.get("category", ""),
        "difficulty_level":     project_data.get("difficulty_level", ""),
        "team_members":         project_data.get("team_members", []),
        "required_skills":      project_data.get("required_skills", []),
        "project_budget":       project_data.get("project_budget", ""),
        "estimated_duration":   project_data.get("estimated_duration", ""),
        "application_deadline": project_data.get("application_deadline", ""),
        "milestone_plans":      [],
    }

    final_state = _graph.invoke(initial_state)

    return {
        "milestone_plans": final_state["milestone_plans"],
    }
