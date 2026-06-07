import asyncio
import uuid
from litellm import acompletion
from helios.generators.synthetic_data import generate_test_cases_from_chunk
from helios.metrics.judge import generate_eval_prompt, parse_llm_response
from helios.storage.database import init_db, save_test_cases, save_evaluation_result
from helios.runners.async_runner import rate_limit

async def run_full_evaluation_pipeline():
    print("=== INITIALIZING HELIOS EVALUATION ENGINE ===\n")
    
    init_db()
    current_run_id = f"run_{uuid.uuid4().hex[:8]}"
    print(f"[*] Started Run: {current_run_id}")

    corpus_chunk = """
    Aegis Corp Server Deployment Guide:
    Production servers must use Ubuntu 22.04 LTS. 
    Deployments are scheduled strictly for Tuesdays at 2:00 AM EST to minimize user disruption.
    Emergency hotfixes require approval from two Lead DevOps Engineers and bypass the Tuesday window.
    """

    print("[*] Generating synthetic test cases...")
    dataset = await generate_test_cases_from_chunk(corpus_chunk, num_cases=2)
    
    if not dataset:
        print("Failed to generate dataset. Aborting.")
        return

    save_test_cases(current_run_id, "mock_rag_system", dataset.test_cases)
    
    print("\n[*] Firing Test Cases through Judge...")
    
    for idx, case in enumerate(dataset.test_cases, 1):
        print(f"\nEvaluating Case {idx} [{case.difficulty}]")
        
        simulated_bad_answer = f"According to the guide, deployments happen on Wednesdays. {case.ground_truth}"
        prompt = generate_eval_prompt(case.question, corpus_chunk, simulated_bad_answer)

        try:
            async with rate_limit:
                response = await acompletion(
                    model="gemini/gemini-2.5-flash",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    num_retries=3
                )
                
                raw_output = response.choices[0].message.content
                metric = parse_llm_response(raw_output)
                
                if metric:
                    print(f" -> Score: {metric.score}/5 | Hallucinated: {metric.hallucination_flag}")
                    save_evaluation_result(idx, simulated_bad_answer, metric.score, metric.hallucination_flag, metric.reasoning)
                    
        except Exception as e:
            print(f" -> ERROR: Judge failed on Case {idx} due to network timeout. Skipping to next case.")
            print(f" -> System Diagnostic: {type(e).__name__}")

    print("\n=== PIPELINE COMPLETE. DATA SECURED IN SQLITE. ===")

if __name__ == "__main__":
    asyncio.run(run_full_evaluation_pipeline())