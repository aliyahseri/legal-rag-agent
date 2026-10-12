"""
Test cases for legal agent evaluation.
Phase 8: EvalCase dataclass and test case definitions.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class EvalCase:
    """Evaluation test case for legal agent."""
    case_id: str
    case_name: str
    query: str
    expected_topics: List[str]
    expected_sources_contain: List[str]
    min_citations: int


# 10 test cases covering different legal topics
TEST_CASES: List[EvalCase] = [
    EvalCase(
        case_id="case_1",
        case_name="Force Majeure - COVID-19",
        query="Does COVID-19 qualify as a force majeure event?",
        expected_topics=["force majeure", "COVID-19", "pandemic", "WHO"],
        expected_sources_contain=["force_majeure"],
        min_citations=1
    ),
    EvalCase(
        case_id="case_2",
        case_name="Non-Compete Agreement",
        query="Are non-compete agreements enforceable in California?",
        expected_topics=["non-compete", "California", "Business and Professions Code", "Section 16600"],
        expected_sources_contain=["non_compete"],
        min_citations=1
    ),
    EvalCase(
        case_id="case_3",
        case_name="GDPR Compliance",
        query="What are the data subject rights under GDPR?",
        expected_topics=["GDPR", "data subject rights", "access", "rectification", "erasure"],
        expected_sources_contain=["gdpr"],
        min_citations=1
    ),
    EvalCase(
        case_id="case_4",
        case_name="IP Assignment",
        query="What rights are assigned in an IP assignment agreement?",
        expected_topics=["IP assignment", "patents", "copyrights", "trademarks", "trade secrets"],
        expected_sources_contain=["ip_assignment"],
        min_citations=1
    ),
    EvalCase(
        case_id="case_5",
        case_name="Liquidated Damages",
        query="What is the difference between liquidated damages and penalties?",
        expected_topics=["liquidated damages", "penalty", "reasonable", "pre-estimate of loss"],
        expected_sources_contain=["liquidated_damages"],
        min_citations=1
    ),
    EvalCase(
        case_id="case_6",
        case_name="Auto-Renewal Clause",
        query="What are the requirements for a valid auto-renewal clause?",
        expected_topics=["auto-renewal", "notice", "termination", "contract"],
        expected_sources_contain=["contract"],
        min_citations=1
    ),
    EvalCase(
        case_id="case_7",
        case_name="Whistleblower Protection",
        query="What protections do whistleblowers have under federal law?",
        expected_topics=["whistleblower", "protection", "Sarbanes-Oxley", "Dodd-Frank", "retaliation"],
        expected_sources_contain=["case_law"],
        min_citations=1
    ),
    EvalCase(
        case_id="case_8",
        case_name="Construction Contracts",
        query="What are the key provisions in a construction contract?",
        expected_topics=["construction", "contract", "scope", "payment", "termination"],
        expected_sources_contain=["contract"],
        min_citations=1
    ),
    EvalCase(
        case_id="case_9",
        case_name="Trade Secrets",
        query="What constitutes a trade secret under the Defend Trade Secrets Act?",
        expected_topics=["trade secrets", "DTSA", "18 U.S.C. § 1836", "misappropriation"],
        expected_sources_contain=["ip_assignment", "trade_secrets"],
        min_citations=1
    ),
    EvalCase(
        case_id="case_10",
        case_name="Arbitration Clause",
        query="What are the enforceability requirements for arbitration clauses?",
        expected_topics=["arbitration", "enforceability", "Federal Arbitration Act", "agreement"],
        expected_sources_contain=["case_law", "contract"],
        min_citations=1
    )
]
