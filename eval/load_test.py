"""Locust load test for the deployed juris-rag endpoint.

Simulates the interview-spec traffic: 100 requests/second sustained.
Retrieval-only mode dominates because it exercises the model layer without
Gemini spend; a small share of full /ask calls models the real mix.

Run against a deployed ALB:
    locust -f eval/load_test.py --host http://<alb-dns> --users 50 --spawn-rate 5

Sanity-check locally against the container:
    docker build -t juris-rag . && docker run -p 8080:8080 juris-rag
    locust -f eval/load_test.py --host http://localhost:8080
"""

from __future__ import annotations

import random

from locust import HttpUser, between, task

QUESTIONS = [
    "How does the three-judge tribunal work?",
    "Where are judge model IDs stored for runtime changes?",
    "How does PANCH defend against prompt injection?",
    "What schema must judge outputs conform to?",
    "What happens when judges disagree about a ruling?",
    "How is the frontend deployed?",
    "What is the escrow and on-chain settlement plan?",
    "What are the cost-tracking controls?",
]


class RagUser(HttpUser):
    wait_time = between(0.5, 1.5)

    @task(9)
    def retrieval(self) -> None:
        self.client.post(
            "/ask/retrieval",
            json={"question": random.choice(QUESTIONS), "k": 5},
        )

    @task(1)
    def full_answer(self) -> None:
        self.client.post(
            "/ask",
            json={"question": random.choice(QUESTIONS), "k": 5},
        )

    @task(1)
    def health(self) -> None:
        self.client.get("/health")
