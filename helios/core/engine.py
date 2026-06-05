import asyncio
import os
from dotenv import load_dotenv
from litellm import acompletion

from helios.metrics.judge import generate_eval_prompt, parse_llm_response

load_dotenv()

async def run_live_evaluation(question: str, context: str, answer: str) -> None:

    prompt = generate_eval_prompt(question, context, answer)

    model_routing = "gemini/gemini-2.5-flash" 
    
    print(f"Connecting to remote infrastructure via LiteLLM [{model_routing}]...")
    
    try:
        response = await acompletion(
            model=model_routing,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        
        raw_output = response.choices[0].message.content
        print("\n RAW NETWORK RESPONSE RECEIVED")
        print(raw_output)

        print("\n RUNNING PYDANTIC VALIDATION CONTRACT")
        metric_result = parse_llm_response(raw_output)
        
        if metric_result:
            print("Successfully compiled live evaluation metadata!")
            print(f"Judged Score: {metric_result.score}/5")
            print(f"Hallucination Flag: {metric_result.hallucination_flag}")
            print(f"Explanation: {metric_result.reasoning}")
            
    except Exception as e:
        print(f"Network or Authentication Error: {e}")

if __name__ == "__main__":
    # Test Case: We deliberately pass an answer that hallucinates numbers
    test_q = "What was our quarterly net profit growth?"
    test_c = "The fiscal report states net profit grew by 14% quarter-over-quarter."
    test_a = "Our net profit skyrocketed by a massive 40% this quarter!"
    
    # Fire the event loop
    asyncio.run(run_live_evaluation(test_q, test_c, test_a))