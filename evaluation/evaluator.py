"""
Legal agent evaluator with RAGAS metrics.
Phase 8: Faithfulness, task success, latency, and cost evaluation.
"""

import time
import re
from typing import List, Dict
from dataclasses import dataclass
import pandas as pd
from langchain_core.documents import Document
from llm.llm_factory import get_llm
from rich.console import Console
from rich.table import Table
from evaluation.test_cases import EvalCase

console = Console()


class LegalAgentEvaluator:
    """Evaluate legal agent performance across multiple metrics."""
    
    def __init__(self):
        """
        Initialize the evaluator.
        """
        self.llm = get_llm()
    
    def measure_faithfulness(self, answer: str, context_docs: List[Document]) -> float:
        """
        Measure faithfulness using RAGAS-style metric.
        
        Args:
            answer: Generated answer from the agent.
            context_docs: Retrieved context documents.
            
        Returns:
            Faithfulness score between 0.0 and 1.0.
        """
        if not context_docs:
            return 0.0
        
        # Format context
        context = "\n\n".join([doc.page_content for doc in context_docs])
        
        # Extract claims from answer (simple sentence-based extraction)
        claims = re.split(r'[.!?]', answer)
        claims = [c.strip() for c in claims if c.strip()]
        
        if not claims:
            return 0.0
        
        # Check each claim against context
        supported_claims = 0
        for claim in claims:
            # Simple keyword-based check (in production, use more sophisticated NLI)
            claim_words = set(claim.lower().split())
            context_words = set(context.lower().split())
            
            # If significant overlap, consider supported
            overlap = len(claim_words & context_words)
            if overlap >= len(claim_words) * 0.3:
                supported_claims += 1
        
        faithfulness = supported_claims / len(claims)
        return round(faithfulness, 2)
    
    def measure_task_success(self, answer: str, eval_case: EvalCase) -> Dict[str, any]:
        """
        Measure task success based on expected topics and citations.
        
        Args:
            answer: Generated answer from the agent.
            eval_case: Evaluation test case.
            
        Returns:
            Dictionary with success status, score, and failures.
        """
        failures = []
        answer_lower = answer.lower()
        
        # Check expected topics
        for topic in eval_case.expected_topics:
            if topic.lower() not in answer_lower:
                failures.append(f"Missing topic: {topic}")
        
        # Check expected sources
        for source in eval_case.expected_sources_contain:
            if source.lower() not in answer_lower:
                failures.append(f"Missing source reference: {source}")
        
        # Check minimum citations
        citation_pattern = r'\[Source: [^]]+\]'
        citations = re.findall(citation_pattern, answer)
        if len(citations) < eval_case.min_citations:
            failures.append(f"Insufficient citations: {len(citations)} < {eval_case.min_citations}")
        
        # Check disclaimer
        if "consult a licensed attorney" not in answer_lower:
            failures.append("Missing disclaimer")
        
        # Check answer length
        if len(answer) < 100:
            failures.append("Answer too short")
        
        success = len(failures) == 0
        score = 1.0 if success else max(0.0, 1.0 - (len(failures) * 0.2))
        
        return {
            "success": success,
            "score": round(score, 2),
            "failures": failures,
            "citations_found": len(citations)
        }
    
    def measure_latency(self, agent, question: str, runs: int = 3) -> Dict[str, float]:
        """
        Measure latency across multiple runs.
        
        Args:
            agent: Agent instance to test.
            question: Test question.
            runs: Number of runs to average over.
            
        Returns:
            Dictionary with mean, min, and max latency in seconds.
        """
        latencies = []
        
        for i in range(runs):
            start_time = time.time()
            try:
                agent.run(question)
            except Exception as e:
                console.print(f"[yellow]Run {i+1} failed: {e}[/yellow]")
                continue
            end_time = time.time()
            latencies.append(end_time - start_time)
        
        if not latencies:
            return {"mean": 0.0, "min": 0.0, "max": 0.0}
        
        return {
            "mean": round(sum(latencies) / len(latencies), 2),
            "min": round(min(latencies), 2),
            "max": round(max(latencies), 2)
        }
    
    def measure_cost(self, question: str, answer: str) -> Dict[str, any]:
        """
        Estimate token usage and cost for GPT-4o.
        
        Args:
            question: Input question.
            answer: Generated answer.
            
        Returns:
            Dictionary with token counts and estimated cost.
        """
        # Simple token estimation (rough approximation)
        input_tokens = len(question.split()) + len(answer.split()) * 0.5
        output_tokens = len(answer.split())
        
        # GPT-4o pricing (as of 2024)
        # Input: $2.50 per 1M tokens
        # Output: $10.00 per 1M tokens
        input_cost = (input_tokens / 1_000_000) * 2.50
        output_cost = (output_tokens / 1_000_000) * 10.00
        total_cost = input_cost + output_cost
        
        return {
            "input_tokens": int(input_tokens),
            "output_tokens": int(output_tokens),
            "cost_usd": round(total_cost, 6)
        }
    
    def run_full_evaluation(self, agent, test_cases: List[EvalCase], 
                           context_docs_map: Dict[str, List[Document]] = None) -> pd.DataFrame:
        """
        Run full evaluation suite on all test cases.
        
        Args:
            agent: Agent instance to evaluate.
            test_cases: List of evaluation test cases.
            context_docs_map: Optional mapping of case_id to context docs.
            
        Returns:
            DataFrame with evaluation results.
        """
        results = []
        
        console.print("\n[blue]Running full evaluation suite...[/blue]\n")
        
        for eval_case in test_cases:
            console.print(f"[cyan]Evaluating: {eval_case.case_name}[/cyan]")
            
            # Run agent
            try:
                result = agent.run(eval_case.query)
                answer = result.get("answer", "")
                success = result.get("success", False)
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
                answer = ""
                success = False
            
            # Get context docs if available
            context_docs = context_docs_map.get(eval_case.case_id, []) if context_docs_map else []
            
            # Measure metrics
            faithfulness = self.measure_faithfulness(answer, context_docs)
            task_success = self.measure_task_success(answer, eval_case)
            latency = self.measure_latency(agent, eval_case.query, runs=1)
            cost = self.measure_cost(eval_case.query, answer)
            
            # Calculate overall score
            overall_score = (faithfulness * 0.4 + task_success["score"] * 0.4 + 
                            (1 - min(latency["mean"] / 10, 1)) * 0.1 + 
                            (1 - min(cost["cost_usd"] / 0.01, 1)) * 0.1)
            
            results.append({
                "case_id": eval_case.case_id,
                "case_name": eval_case.case_name,
                "faithfulness": faithfulness,
                "task_success": task_success["score"],
                "task_failures": task_success["failures"],
                "latency_mean": latency["mean"],
                "cost_usd": cost["cost_usd"],
                "overall_score": round(overall_score, 2)
            })
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Print rich table
        self._print_results_table(df)
        
        return df
    
    def _print_results_table(self, df: pd.DataFrame):
        """Print evaluation results as a rich table."""
        table = Table(title="Legal Agent Evaluation Results")
        
        table.add_column("Case ID", style="cyan")
        table.add_column("Case Name", style="magenta")
        table.add_column("Faithfulness", justify="right")
        table.add_column("Task Success", justify="right")
        table.add_column("Latency (s)", justify="right")
        table.add_column("Cost ($)", justify="right")
        table.add_column("Overall", justify="right", style="green")
        
        for _, row in df.iterrows():
            table.add_row(
                row["case_id"],
                row["case_name"],
                f"{row['faithfulness']:.2f}",
                f"{row['task_success']:.2f}",
                f"{row['latency_mean']:.2f}",
                f"{row['cost_usd']:.6f}",
                f"{row['overall_score']:.2f}"
            )
        
        console.print(table)
        
        # Print summary
        console.print(f"\n[green]Average Overall Score: {df['overall_score'].mean():.2f}[/green]")
        console.print(f"[green]Average Faithfulness: {df['faithfulness'].mean():.2f}[/green]")
        console.print(f"[green]Average Task Success: {df['task_success'].mean():.2f}[/green]")
    
    def compare_models(self, question: str, models: List[str]) -> pd.DataFrame:
        """
        Compare different models on the same question.
        
        Args:
            question: Test question.
            models: List of model names to compare.
            
        Returns:
            DataFrame with comparison results.
        """
        results = []
        
        console.print(f"\n[blue]Comparing models on question: {question}[/blue]\n")
        console.print("[yellow]Note: Using default LLM from factory for all comparisons[/yellow]")
        
        # Use the default LLM from factory for comparison
        llm = get_llm()
        
        for model in models:
            console.print(f"[cyan]Testing model: {model}[/cyan]")
            
            # Simple quality check (in production, use human evaluation)
            try:
                response = llm.invoke(f"Answer this legal question: {question}")
                answer = response.content
                
                # Estimate quality based on length and structure
                quality_score = min(len(answer) / 500, 1.0)
                
                # Estimate latency
                start = time.time()
                llm.invoke(f"Answer this legal question: {question}")
                latency = time.time() - start
                
                # Estimate cost
                cost = self.measure_cost(question, answer)
                
                results.append({
                    "model": model,
                    "quality_score": round(quality_score, 2),
                    "latency_s": round(latency, 2),
                    "cost_usd": cost["cost_usd"]
                })
                
            except Exception as e:
                console.print(f"[red]Error with {model}: {e}[/red]")
                results.append({
                    "model": model,
                    "quality_score": 0.0,
                    "latency_s": 0.0,
                    "cost_usd": 0.0
                })
        
        df = pd.DataFrame(results)
        
        # Print comparison table
        table = Table(title="Model Comparison Results")
        table.add_column("Model", style="cyan")
        table.add_column("Quality", justify="right")
        table.add_column("Latency (s)", justify="right")
        table.add_column("Cost ($)", justify="right")
        
        for _, row in df.iterrows():
            table.add_row(
                row["model"],
                f"{row['quality_score']:.2f}",
                f"{row['latency_s']:.2f}",
                f"{row['cost_usd']:.6f}"
            )
        
        console.print(table)
        
        return df
