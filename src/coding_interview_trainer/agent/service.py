from time import perf_counter

from coding_interview_trainer.agent.llm import LLMBackend, make_backend
from coding_interview_trainer.prompts.templates import build_prompt
from coding_interview_trainer.schemas import AnalyzeResponse, InterviewRequest


def parse_sections(text: str) -> dict[str, str]:
    sections: dict[str, list[str]] = {}
    current: str | None = None

    for line in text.splitlines():
        if line.startswith("## "):
            current = line[3:].strip()
            sections[current] = []
        elif current:
            sections[current].append(line)

    return {key: "\n".join(value).strip() for key, value in sections.items()}


class InterviewAgent:
    def __init__(self, backend: LLMBackend | None = None) -> None:
        self.backend = backend or make_backend()

    def analyze(self, request: InterviewRequest) -> AnalyzeResponse:
        prompt = build_prompt(request)
        started = perf_counter()
        raw_response = self.backend.generate(prompt)
        latency_ms = (perf_counter() - started) * 1000
        return AnalyzeResponse(
            mode=request.mode,
            model_name=self.backend.model_name,
            backend=self.backend.backend_name,
            prompt=prompt,
            raw_response=raw_response,
            sections=parse_sections(raw_response),
            latency_ms=latency_ms,
        )
