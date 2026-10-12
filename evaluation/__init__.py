"""
Evaluation module for Legal RAG Assistant.
Phase 8: Test cases and evaluator implementation.
"""

from .test_cases import EvalCase, TEST_CASES
from .evaluator import LegalAgentEvaluator

__all__ = ["EvalCase", "TEST_CASES", "LegalAgentEvaluator"]
