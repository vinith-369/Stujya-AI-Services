from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Optional
from llm_service import generate_milestone_plans

router = APIRouter(
    prefix="/milestones",
    tags=["Milestones"],
)


# ---------------------------------------------------------------------------
# Request Schema
# ---------------------------------------------------------------------------

class MilestoneRequest(BaseModel):
    """
    Project details collected from the project creation form.
    Team members are represented only by their COUNT (team_size).
    The AI will assign appropriate roles automatically.
    """
    project_title: str = Field(
        ...,
        example="E-commerce Mobile App Development",
        description="The title of the project.",
    )
    short_description: str = Field(
        ...,
        example="A brief overview of the project (100–150 words).",
        description="Short summary of the project.",
    )
    detailed_description: str = Field(
        ...,
        example="Detailed requirements, goals, and context for the project.",
        description="In-depth description including goals and technical context.",
    )
    category: str = Field(
        ...,
        example="Mobile Development",
        description="Project category (e.g., Web, AI/ML, Mobile).",
    )
    difficulty_level: str = Field(
        ...,
        example="Intermediate",
        description="Difficulty level: Beginner, Intermediate, or Advanced.",
    )
    team_size: int = Field(
        ...,
        ge=1,
        le=20,
        example=4,
        description="Total number of team members (used to auto-assign roles).",
    )
    required_skills: List[str] = Field(
        ...,
        example=["React Native", "Node.js", "PostgreSQL", "REST APIs"],
        description="List of skills required for the project.",
    )
    project_budget: Optional[str] = Field(
        None,
        example="$5,000",
        description="Estimated project budget.",
    )
    estimated_duration: Optional[str] = Field(
        None,
        example="2 months",
        description="Expected duration of the project.",
    )
    application_deadline: Optional[str] = Field(
        None,
        example="2026-05-01",
        description="Application deadline (ISO date or readable string).",
    )


class MilestoneItem(BaseModel):
    milestone_number: int
    title: str
    description: str
    due_date: str
    role: str


class MemberMilestonePlan(BaseModel):
    member: str          # e.g. "Member 1"
    role: str            # e.g. "Frontend Developer"
    milestones: List[MilestoneItem]


class AssignedRole(BaseModel):
    member: str
    role: str


class MilestoneResponse(BaseModel):
    success: bool
    project_title: str
    team_size: int
    assigned_roles: List[AssignedRole]
    milestone_plans: List[MemberMilestonePlan]


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post(
    "/generate",
    response_model=MilestoneResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate per-member milestone plans",
    description=(
        "Accepts project details and team size. "
        "The AI first auto-assigns a role to each member slot based on the "
        "project's domain and required skills, then generates an individualised "
        "4–6 milestone plan for each role."
    ),
)
async def generate_milestones(payload: MilestoneRequest):
    """
    **POST /milestones/generate**

    Two-step AI pipeline (LangGraph):
    1. **assign_roles** — Determines the best role for each of the `team_size` slots.
    2. **generate_milestones** — Produces role-specific milestones for each member.

    Each milestone includes: `milestone_number`, `title`, `description`,
    `due_date`, and `role`.
    """
    try:
        result = generate_milestone_plans(payload.model_dump())
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"LLM returned unparseable output: {ve}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Milestone generation failed: {e}",
        )

    return MilestoneResponse(
        success=True,
        project_title=payload.project_title,
        team_size=payload.team_size,
        assigned_roles=result.get("assigned_roles", []),
        milestone_plans=result.get("milestone_plans", []),
    )
