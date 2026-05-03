from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from milestone_router import router as milestone_router

app = FastAPI(
    title="Stujya AI API",
    description="AI-powered milestone planning for student projects.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(milestone_router)


@app.get("/", tags=["Health"])
async def root():
    return {"status": "ok", "message": "Stujya AI API is running."}
