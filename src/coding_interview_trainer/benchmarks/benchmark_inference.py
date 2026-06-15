from __future__ import annotations

import argparse
import json
from statistics import mean
from time import perf_counter

from coding_interview_trainer.agent.service import InterviewAgent
from coding_interview_trainer.schemas import AssistanceMode, InterviewRequest


def sample_request() -> InterviewRequest:
    return InterviewRequest(
        problem_title="Longest Substring Without Repeating Characters",
        problem_description="Given a string s, return the length of the longest substring without repeating characters.",
        programming_language="Python",
        mode=AssistanceMode.BUG_DIAGNOSIS,
        attempted_solution="def lengthOfLongestSubstring(s):\n    seen = {}\n    left = 0\n    best = 0\n    for right, ch in enumerate(s):\n        if ch in seen:\n            left = seen[ch] + 1\n        seen[ch] = right\n        best = max(best, right - left + 1)\n    return best",
        failing_test_case='s = "abba"',
        expected_output="2",
        wrong_output="3",
    )


def benchmark(iterations: int) -> dict[str, float | int | str]:
    agent = InterviewAgent()
    latencies = []
    output_lengths = []
    for _ in range(iterations):
        started = perf_counter()
        response = agent.analyze(sample_request())
        latencies.append((perf_counter() - started) * 1000)
        output_lengths.append(len(response.raw_response))
    return {
        "backend": agent.backend.backend_name,
        "model_name": agent.backend.model_name,
        "iterations": iterations,
        "average_latency_ms": mean(latencies),
        "min_latency_ms": min(latencies),
        "max_latency_ms": max(latencies),
        "average_output_chars": mean(output_lengths),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the configured inference backend.")
    parser.add_argument("--iterations", type=int, default=3)
    args = parser.parse_args()
    print(json.dumps(benchmark(args.iterations), indent=2))


if __name__ == "__main__":
    main()
