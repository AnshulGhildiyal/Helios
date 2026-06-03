import asyncio
import time
from aiolimiter import AsyncLimiter

rate_limit = AsyncLimiter(5, 1)

async def mock_llm_call(task_id: int) -> str:
    async with rate_limit:
        print(f"[{time.strftime('%X')}] Firing API request for Test Case {task_id}...")
        await asyncio.sleep(0.5) 
        
        return f"Result_{task_id}"

async def run_evaluations(num_tasks: int):
    print(f"Starting {num_tasks} evaluations in parallel...")
    start_time = time.time()

    tasks = [mock_llm_call(i) for i in range(1, num_tasks + 1)]
    results = await asyncio.gather(*tasks)

    elapsed = time.time() - start_time
    print(f"\n--- EVALUATION COMPLETE ---")
    print(f"Processed {len(results)} cases in {elapsed:.2f} seconds.")
    print(f"Throughput: {len(results) / elapsed:.2f} requests/second")

if __name__ == "__main__":
    asyncio.run(run_evaluations(15))