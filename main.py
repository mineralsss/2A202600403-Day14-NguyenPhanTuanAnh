import asyncio
import json
import os
import time
from engine.runner import BenchmarkRunner
from engine.retrieval_eval import RetrievalEvaluator
from engine.llm_judge import LLMJudge
from agent.main_agent import MainAgent

class ExpertEvaluator:
    def __init__(self):
        self.retrieval_eval = RetrievalEvaluator()

    async def score(self, case, resp):
        # Real retrieval metrics
        expected_src = os.path.basename(case.get("metadata", {}).get("source", ""))
        retrieved_srcs = [os.path.basename(s) for s in resp.get("metadata", {}).get("sources", [])]
        hit_rate = self.retrieval_eval.calculate_hit_rate([expected_src], retrieved_srcs)
        mrr = self.retrieval_eval.calculate_mrr([expected_src], retrieved_srcs)

        # Simple faithfulness: word overlap between expected and actual answer
        expected_words = set(case.get("expected_answer", "").lower().split())
        actual_words = set(resp.get("answer", "").lower().split())
        faithfulness = len(expected_words & actual_words) / max(len(expected_words), 1)

        # Simple relevancy: question-term overlap with retrieved contexts
        q_words = set(case.get("question", "").lower().split())
        ctx_words = set(" ".join(resp.get("contexts", [])).lower().split())
        relevancy = len(q_words & ctx_words) / max(len(q_words), 1)

        return {
            "faithfulness": round(min(faithfulness, 1.0), 3),
            "relevancy": round(min(relevancy, 1.0), 3),
            "retrieval": {"hit_rate": hit_rate, "mrr": mrr}
        }

V2_SYSTEM_PROMPT = (
    "Bạn là trợ lý AI chuyên nghiệp. Hãy trả lời câu hỏi dựa trên các đoạn tài liệu được cung cấp. "
    "Yêu cầu: (1) Trả lời chính xác và đầy đủ, (2) Trích dẫn nguồn tài liệu khi có thể, "
    "(3) Nếu không đủ thông tin, nói rõ phần nào thiếu thay vì đoán."
)

async def run_benchmark_with_results(agent_version: str, agent: MainAgent = None):
    print(f"🚀 Khởi động Benchmark cho {agent_version}...")

    if not os.path.exists("data/golden_set.jsonl"):
        print("❌ Thiếu data/golden_set.jsonl. Hãy chạy 'python data/synthetic_gen.py' trước.")
        return None, None

    with open("data/golden_set.jsonl", "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]

    if not dataset:
        print("❌ File data/golden_set.jsonl rỗng. Hãy tạo ít nhất 1 test case.")
        return None, None

    if agent is None:
        agent = MainAgent()

    runner = BenchmarkRunner(agent, ExpertEvaluator(), LLMJudge())
    results = await runner.run_all(dataset)

    total = len(results)
    summary = {
        "metadata": {"version": agent_version, "total": total, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")},
        "metrics": {
            "avg_score": sum(r["judge"]["final_score"] for r in results) / total,
            "hit_rate": sum(r["ragas"]["retrieval"]["hit_rate"] for r in results) / total,
            "agreement_rate": sum(r["judge"]["agreement_rate"] for r in results) / total
        }
    }
    return results, summary

async def run_benchmark(version, agent=None):
    _, summary = await run_benchmark_with_results(version, agent)
    return summary

# Set to True to use local Qwen3.5-4B-Base pipeline instead of OpenAI for V1
USE_QWEN = True

async def main():
    # V1: baseline
    if USE_QWEN:
        # Qwen3.5-4B-Base local pipeline with Qwen embeddings
        print("🧠 Using Qwen3.5-4B-Base pipeline for V1...")
        agent_v1 = MainAgent(top_k=2, use_qwen=True)
    else:
        # OpenRouter cloud model
        agent_v1 = MainAgent(top_k=2, model_name="z-ai/glm-4.5-air:free", openrouter=True)
    v1_summary = await run_benchmark("Agent_V1_Base", agent_v1)
    
    # V2: improved — gpt-4o-mini, top_k=3, better prompt with citation instructions
    agent_v2 = MainAgent(top_k=3, system_prompt=V2_SYSTEM_PROMPT)
    v2_results, v2_summary = await run_benchmark_with_results("Agent_V2_Optimized", agent_v2)
    
    if not v1_summary or not v2_summary:
        print("❌ Không thể chạy Benchmark. Kiểm tra lại data/golden_set.jsonl.")
        return

    print("\n📊 --- KẾT QUẢ SO SÁNH (REGRESSION) ---")
    delta = v2_summary["metrics"]["avg_score"] - v1_summary["metrics"]["avg_score"]
    print(f"V1 Score: {v1_summary['metrics']['avg_score']}")
    print(f"V2 Score: {v2_summary['metrics']['avg_score']}")
    print(f"Delta: {'+' if delta >= 0 else ''}{delta:.2f}")

    os.makedirs("reports", exist_ok=True)
    with open("reports/summary.json", "w", encoding="utf-8") as f:
        json.dump(v2_summary, f, ensure_ascii=False, indent=2)
    with open("reports/benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(v2_results, f, ensure_ascii=False, indent=2)

    if delta > 0:
        print("✅ QUYẾT ĐỊNH: CHẤP NHẬN BẢN CẬP NHẬT (APPROVE)")
    else:
        print("❌ QUYẾT ĐỊNH: TỪ CHỐI (BLOCK RELEASE)")

if __name__ == "__main__":
    asyncio.run(main())
