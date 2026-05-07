"""
main.py
-------
Standalone runner for the milestone generator.
No web server — just call generate_milestone_plans() and print the result.

Usage:
    python main.py
"""

import json
from llm_service import generate_milestone_plans

# ---------------------------------------------------------------------------
# Sample project payload  (edit as needed)
# ---------------------------------------------------------------------------

project_data = {
    "project_title": "E-commerce Mobile App Development",
    "short_description": (
        "A cross-platform mobile app that lets users browse products, "
        "manage carts, and complete purchases via multiple payment gateways."
    ),
    "detailed_description": (
        "The app will be built with React Native for iOS/Android, backed by "
        "a Node.js REST API, PostgreSQL for persistent storage, and Redis "
        "for session caching. It will include push notifications, order "
        "tracking, and an admin dashboard."
    ),
    "category": "Mobile Development",
    "difficulty_level": "Intermediate",
    "required_skills": ["React Native", "Node.js", "PostgreSQL", "REST APIs", "UI/UX Design"],
    "project_budget": "$5,000",
    "estimated_duration": "2 months",
    "application_deadline": "2026-06-01",

    # -----------------------------------------------------------------------
    # team_members — any shape is accepted.
    # Pass whatever info you have: name, skills, role, experience, etc.
    # -----------------------------------------------------------------------
    "team_members": [
        {
            "name": "Alice",
            "specialisation": "Frontend / React Native",
            "experience_years": 2,
            "skills": ["React Native", "TypeScript", "Figma"],
        },
        {
            "name": "Bob",
            "specialisation": "Backend / Node.js",
            "skills": ["Node.js", "Express", "PostgreSQL", "Redis"],
        },
        {
            "name": "Carol",
            "specialisation": "UI/UX Designer",
            "skills": ["Figma", "User Research", "Prototyping", "CSS"],
        },
        {
            "name": "Dave",
            "specialisation": "QA / DevOps",
            "skills": ["Jest", "Cypress", "Docker", "GitHub Actions"],
        },
    ],
}

# ---------------------------------------------------------------------------
# Run the pipeline and print output
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Generating milestone plans...\n")
    result = generate_milestone_plans(project_data)
    print(json.dumps(result, indent=2))
