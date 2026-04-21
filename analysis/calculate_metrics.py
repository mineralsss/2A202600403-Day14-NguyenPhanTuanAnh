"""
Script tính toán Faithfulness, Relevancy và các metrics khác
từ reports/benchmark_results.json
"""
import json
import os
import sys
import io

# Force utf-8 encoding for standard output to avoid charmap errors on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def load_results(path="reports/benchmark_results.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def calculate_metrics(results):
    n = len(results)
    
    # RAGAS metrics
    faithfulness_scores = [r["ragas"]["faithfulness"] for r in results]
    relevancy_scores = [r["ragas"]["relevancy"] for r in results]
    hit_rates = [r["ragas"]["retrieval"]["hit_rate"] for r in results]
    mrr_scores = [r["ragas"]["retrieval"]["mrr"] for r in results]
    
    # Judge metrics
    judge_scores = [r["judge"]["final_score"] for r in results]
    agreement_rates = [r["judge"]["agreement_rate"] for r in results]
    
    # Latency
    latencies = [r["latency"] for r in results]
    
    # Pass/Fail
    pass_count = sum(1 for r in results if r["status"] == "pass")
    fail_count = n - pass_count
    
    # Per-judge breakdown
    gpt_accuracy = []
    gpt_tone = []
    gpt_safety = []
    gemma_accuracy = []
    gemma_tone = []
    gemma_safety = []
    
    for r in results:
        scores = r["judge"]["individual_scores"]
        gpt = scores.get("gpt-4o", {})
        gemma = scores.get("gemma-4-26B-A4B-it", {})
        gpt_accuracy.append(gpt.get("accuracy", 3))
        gpt_tone.append(gpt.get("tone", 3))
        gpt_safety.append(gpt.get("safety", 3))
        gemma_accuracy.append(gemma.get("accuracy", 3))
        gemma_tone.append(gemma.get("tone", 3))
        gemma_safety.append(gemma.get("safety", 3))
    
    avg = lambda lst: sum(lst) / len(lst) if lst else 0
    
    metrics = {
        "total_cases": n,
        "pass_count": pass_count,
        "fail_count": fail_count,
        "ragas": {
            "avg_faithfulness": round(avg(faithfulness_scores), 4),
            "avg_relevancy": round(avg(relevancy_scores), 4),
            "avg_hit_rate": round(avg(hit_rates), 4),
            "avg_mrr": round(avg(mrr_scores), 4),
            "min_faithfulness": round(min(faithfulness_scores), 4),
            "max_faithfulness": round(max(faithfulness_scores), 4),
            "min_relevancy": round(min(relevancy_scores), 4),
            "max_relevancy": round(max(relevancy_scores), 4),
        },
        "judge": {
            "avg_final_score": round(avg(judge_scores), 4),
            "avg_agreement_rate": round(avg(agreement_rates), 4),
            "gpt4o": {
                "avg_accuracy": round(avg(gpt_accuracy), 4),
                "avg_tone": round(avg(gpt_tone), 4),
                "avg_safety": round(avg(gpt_safety), 4),
            },
            "gemma": {
                "avg_accuracy": round(avg(gemma_accuracy), 4),
                "avg_tone": round(avg(gemma_tone), 4),
                "avg_safety": round(avg(gemma_safety), 4),
            },
        },
        "latency": {
            "avg": round(avg(latencies), 4),
            "min": round(min(latencies), 4),
            "max": round(max(latencies), 4),
        },
    }
    return metrics

def find_worst_cases(results, n=5):
    """Find the worst performing cases by judge score."""
    sorted_results = sorted(results, key=lambda r: r["judge"]["final_score"])
    return sorted_results[:n]

def find_lowest_faithfulness(results, n=5):
    """Find cases with lowest faithfulness."""
    sorted_results = sorted(results, key=lambda r: r["ragas"]["faithfulness"])
    return sorted_results[:n]

def main():
    results = load_results()
    metrics = calculate_metrics(results)
    
    print("=" * 70)
    print("📊 BENCHMARK METRICS SUMMARY")
    print("=" * 70)
    
    print(f"\n📋 Tổng số cases: {metrics['total_cases']}")
    print(f"✅ Pass: {metrics['pass_count']} | ❌ Fail: {metrics['fail_count']}")
    
    print(f"\n--- RAGAS Metrics ---")
    print(f"  Avg Faithfulness: {metrics['ragas']['avg_faithfulness']:.4f}  (min: {metrics['ragas']['min_faithfulness']}, max: {metrics['ragas']['max_faithfulness']})")
    print(f"  Avg Relevancy:    {metrics['ragas']['avg_relevancy']:.4f}  (min: {metrics['ragas']['min_relevancy']}, max: {metrics['ragas']['max_relevancy']})")
    print(f"  Avg Hit Rate:     {metrics['ragas']['avg_hit_rate']:.4f}")
    print(f"  Avg MRR:          {metrics['ragas']['avg_mrr']:.4f}")
    
    print(f"\n--- LLM Judge Metrics ---")
    print(f"  Avg Final Score:    {metrics['judge']['avg_final_score']:.4f} / 5.0")
    print(f"  Avg Agreement Rate: {metrics['judge']['avg_agreement_rate']:.4f}")
    print(f"  GPT-4o  -> Accuracy: {metrics['judge']['gpt4o']['avg_accuracy']:.2f}, Tone: {metrics['judge']['gpt4o']['avg_tone']:.2f}, Safety: {metrics['judge']['gpt4o']['avg_safety']:.2f}")
    print(f"  Gemma   -> Accuracy: {metrics['judge']['gemma']['avg_accuracy']:.2f}, Tone: {metrics['judge']['gemma']['avg_tone']:.2f}, Safety: {metrics['judge']['gemma']['avg_safety']:.2f}")
    
    print(f"\n--- Latency ---")
    print(f"  Avg: {metrics['latency']['avg']:.2f}s  |  Min: {metrics['latency']['min']:.2f}s  |  Max: {metrics['latency']['max']:.2f}s")
    
    # Worst cases
    print(f"\n{'=' * 70}")
    print("🔴 TOP 5 WORST CASES (by Judge Score)")
    print("=" * 70)
    worst = find_worst_cases(results)
    for i, r in enumerate(worst, 1):
        print(f"\n  #{i}: Score={r['judge']['final_score']:.2f} | F={r['ragas']['faithfulness']:.3f} | R={r['ragas']['relevancy']:.3f}")
        print(f"      Q: {r['test_case'][:80]}...")
        print(f"      A: {r['agent_response'][:100]}...")
    
    print(f"\n{'=' * 70}")
    print("🟡 TOP 5 LOWEST FAITHFULNESS")
    print("=" * 70)
    low_faith = find_lowest_faithfulness(results)
    for i, r in enumerate(low_faith, 1):
        print(f"\n  #{i}: Faithfulness={r['ragas']['faithfulness']:.3f} | Relevancy={r['ragas']['relevancy']:.3f}")
        print(f"      Q: {r['test_case'][:80]}...")
    
    # Save metrics to JSON
    os.makedirs("reports", exist_ok=True)
    with open("reports/detailed_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Detailed metrics saved to reports/detailed_metrics.json")

if __name__ == "__main__":
    main()
