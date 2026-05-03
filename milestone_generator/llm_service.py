"""
llm_service.py
--------------
LangGraph-powered milestone generation pipeline.
Uses Groq API (llama-3.3-70b-versatile) — free tier: 14,400 req/day.

Graph flow:
    START → generate_milestone_plans_node → END

Single node: one LLM call that assigns roles AND generates milestones
for all team members simultaneously.
"""

import os
import json
from typing import TypedDict, List
from dotenv import load_dotenv

load_dotenv()  # loads GEMINI_API_KEY from .env

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
    team_size: int
    required_skills: List[str]
    project_budget: str
    estimated_duration: str
    application_deadline: str
    # Output
    assigned_roles: List[dict]
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


# ---------------------------------------------------------------------------
# Single node: assign roles + generate milestones in ONE LLM call
# ---------------------------------------------------------------------------

def plan_milestones(state: MilestoneState) -> MilestoneState:
    skills_str = ", ".join(state["required_skills"]) if state["required_skills"] else "N/A"

    prompt = f"""
You are an expert project manager planning a student software project.

PROJECT DETAILS:
- Title: {state["project_title"]}
- Short Description: {state["short_description"]}
- Detailed Description: {state["detailed_description"]}
- Category: {state["category"]}
- Difficulty: {state["difficulty_level"]}
- Required Skills: {skills_str}
- Team Size: {state["team_size"]} members
- Budget: {state.get("project_budget") or "N/A"}
- Estimated Duration: {state.get("estimated_duration") or "N/A"}
- Application Deadline: {state.get("application_deadline") or "N/A"}

TASK:
1. Assign one distinct, relevant role to each of the {state["team_size"]} member slots
   (Member 1, Member 2, … Member {state["team_size"]}).
   Roles must suit the project domain and required skills.

2. For each member, generate a personalised milestone plan with 4–6 milestones that:
   - Progress logically: research → design → build → test → deploy/review.
   - Are deeply tailored to that member's role.
   - Fit within the project timeline.

Each milestone must have:
  - milestone_number  (integer, starts at 1 per member)
  - title             (short, action-oriented)
  - description       (2–3 sentences specific to the member's role)
  - due_date          (relative, e.g. "Week 2", "End of Month 1")
  - role              (the role this milestone belongs to)

Return ONLY valid JSON — no markdown, no extra text.

FORMAT:
{{
  "assigned_roles": [
    {{"member": "Member 1", "role": "<role>"}},
    {{"member": "Member 2", "role": "<role>"}}
  ],
  "milestone_plans": [
    {{
      "member": "Member 1",
      "role": "<role>",
      "milestones": [
        {{
          "milestone_number": 1,
          "title": "<title>",
          "description": "<description>",
          "due_date": "<due_date>",
          "role": "<role>"
        }}
      ]
    }}
  ]
}}
""".strip()

    response = llm.invoke([HumanMessage(content=prompt)])
    parsed = _parse_json(response.content)

    state["assigned_roles"] = parsed["assigned_roles"]
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
# Public API
# ---------------------------------------------------------------------------

def generate_milestone_plans(project_data: dict) -> dict:
    """
    Entry point called by the FastAPI router.
    Makes exactly ONE LLM call to assign roles and generate all milestone plans.
    """
    initial_state: MilestoneState = {
        "project_title":        project_data.get("project_title", ""),
        "short_description":    project_data.get("short_description", ""),
        "detailed_description": project_data.get("detailed_description", ""),
        "category":             project_data.get("category", ""),
        "difficulty_level":     project_data.get("difficulty_level", ""),
        "team_size":            int(project_data.get("team_size", 1)),
        "required_skills":      project_data.get("required_skills", []),
        "project_budget":       project_data.get("project_budget", ""),
        "estimated_duration":   project_data.get("estimated_duration", ""),
        "application_deadline": project_data.get("application_deadline", ""),
        "assigned_roles":       [],
        "milestone_plans":      [],
    }

    final_state = _graph.invoke(initial_state)

    return {
        "assigned_roles":  final_state["assigned_roles"],
        "milestone_plans": final_state["milestone_plans"],
    }
