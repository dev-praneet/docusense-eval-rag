import os
from dotenv import load_dotenv
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric, ContextualPrecisionMetric
from deepeval.test_case import LLMTestCase
from deepeval.models import DeepEvalBaseLLM
from llama_index.llms.openai_like import OpenAILike

from rag_engine import initialize_rag_engine

load_dotenv()

# 1. Custom DeepEval LLM Wrapper using Free Groq API
class GroqJudgeLLM(DeepEvalBaseLLM):
    def __init__(self, model_name: str = "openai/gpt-oss-120b"):
        self.model_name = model_name
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY missing from .env file!")
        self.model = OpenAILike(
            model=self.model_name,
            api_base="https://api.groq.com/openai/v1",
            api_key=groq_api_key,
            is_chat_model=True,
            temperature=0.0
        )

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        llm = self.load_model()
        response = llm.complete(prompt)
        return response.text

    async def a_generate(self, prompt: str) -> str:
        llm = self.load_model()
        response = await llm.acomplete(prompt)
        return response.text

    def get_model_name(self):
        return f"Groq ({self.model_name})"


def run_evaluation_suite():
    print("=" * 60)
    print("Starting DocuSense AI Automated Quality Control")
    print("=" * 60)

    # Initialize RAG Pipeline and Judge LLM
    query_engine = initialize_rag_engine()
    eval_llm = GroqJudgeLLM(model_name="openai/gpt-oss-120b")

    # Define 3 Core RAG Metrics (Threshold 0.7 out of 1.0)
    faithfulness = FaithfulnessMetric(threshold=0.7, model=eval_llm, include_reason=True)
    answer_relevancy = AnswerRelevancyMetric(threshold=0.7, model=eval_llm, include_reason=True)
    # context_precision = ContextualPrecisionMetric(threshold=0.7, model=eval_llm, include_reason=True)

    # 5 Test Questions targeting our indexed document
    test_queries = [
        "What is the main algorithm or topic described in the document?",
        "How does the proposed algorithm handle tree traversal or data structures?",
        "What are the key performance considerations mentioned in the text?",
        "Does the paper provide empirical benchmark results or pseudo-code?",
        "What conclusion or summary does the author provide at the end?"
    ]

    results_summary = []

    for idx, query in enumerate(test_queries, 1):
        print(f"\n------------------------------------------------------------")
        print(f"Test Case {idx}/5: '{query}'")
        print(f"------------------------------------------------------------")

        # Execute Query against our RAG engine
        response = query_engine.query(query)
        actual_output = str(response.response)
        retrieval_contexts = [node.node.get_content() for node in response.source_nodes]

        # Construct DeepEval Test Case
        test_case = LLMTestCase(
            input=query,
            actual_output=actual_output,
            retrieval_context=retrieval_contexts
        )

        # Measure Core Metrics
        print("Evaluating Faithfulness (Hallucination Check)...")
        faithfulness.measure(test_case)

        print("Evaluating Answer Relevancy...")
        answer_relevancy.measure(test_case)

        # print("Evaluating Context Precision...")
        # context_precision.measure(test_case)

        # Print Scored Results
        f_score = round(faithfulness.score, 2)
        r_score = round(answer_relevancy.score, 2)
        # p_score = round(context_precision.score, 2)

        print(f"\n Scores for Test Case {idx}:")
        print(f"  • Faithfulness:         {f_score} / 1.0 | Pass: {faithfulness.is_successful()}")
        print(f"  • Answer Relevancy:     {r_score} / 1.0 | Pass: {answer_relevancy.is_successful()}")
        # print(f"  • Context Precision:    {p_score} / 1.0 | Pass: {context_precision.is_successful()}")

        results_summary.append({
            "query": query,
            "faithfulness": f_score,
            "relevancy": r_score,
            # "precision": p_score
        })

    # Print Final Summary Table
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"{'#':<3} | {'Faithfulness':<12} | {'Relevancy':<10}")
    print("-" * 50)
    for i, res in enumerate(results_summary, 1):
        print(f"{i:<3} | {res['faithfulness']:<12} | {res['relevancy']:<10}")
    print("=" * 60)


if __name__ == "__main__":
    run_evaluation_suite()
