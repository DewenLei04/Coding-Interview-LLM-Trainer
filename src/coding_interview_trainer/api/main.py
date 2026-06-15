from functools import lru_cache

from fastapi import FastAPI

from coding_interview_trainer.agent.service import InterviewAgent
from coding_interview_trainer.schemas import AnalyzeResponse, InterviewRequest


app = FastAPI(
    title="Coding Interview LLM Trainer",
    description="Local coding interview tutor API.",
    version="0.1.0",
)


@lru_cache(maxsize=1)
def get_agent() -> InterviewAgent:
    return InterviewAgent()


@app.get("/health")
def health() -> dict[str, str]:
    agent = get_agent()
    return {
        "status": "ok",
        "backend": agent.backend.backend_name,
        "model_name": agent.backend.model_name,
    }


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: InterviewRequest) -> AnalyzeResponse:
    return get_agent().analyze(request)
