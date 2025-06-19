import io
import csv
from datetime import datetime


def format_analysis_for_csv(input_text: str, result: dict) -> dict:
    hate_speech = result.get("hate_speech", {})
    action = result.get("action", {})
    policies = result.get("policies", [])

    policy_sources = [p.get("source", "") for p in policies]
    policy_summaries = [p.get("summary", "") for p in policies]
    policy_scores = [str(p.get("relevance_score", "")) for p in policies]

    return {
        "timestamp": datetime.now().isoformat(),
        "input_text": input_text,
        "classification": hate_speech.get("classification", ""),
        "confidence": hate_speech.get("confidence", ""),
        "classification_reason": hate_speech.get("reason", ""),
        "reasoning_summary": result.get("reasoning", ""),
        "recommended_action": action.get("action", ""),
        "action_severity": action.get("severity", ""),
        "action_reasoning": action.get("reasoning", ""),
        "policy_sources": "; ".join(policy_sources),
        "policy_scores": "; ".join(policy_scores),
        "policy_summaries": "\n\n---\n\n".join(policy_summaries)
    }


def create_csv_buffer(records: list[dict]) -> io.StringIO:
    if not records:
        return io.StringIO()
    headers = list(records[0].keys())
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=headers)
    writer.writeheader()
    writer.writerows(records)
    buffer.seek(0)
    return buffer


def generate_filename(prefix: str = "analysis") -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.csv"
