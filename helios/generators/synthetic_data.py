import asyncio
import json
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv
from litellm import acompletion

load_dotenv()

class TestCase(BaseModel):
    question: str = Field(description="A realistic question a user would ask based on the context.")
    ground_truth: str = Field(description="The factual, correct answer extracted ONLY from the context.")
    difficulty: str = Field(description="Classify as 'factual', 'inferential', or 'adversarial'.")

class TestCaseList(BaseModel):
    test_cases: list[TestCase] = Field(description="A list of generated test cases.")


async def generate_test_cases_from_chunk(raw_text: str, num_cases: int = 3) -> TestCaseList:
    """Takes raw text and generates synthetic QA pairs for evaluation."""
    
    schema_instructions = TestCaseList.model_json_schema()
    
    prompt = f"""
    You are an expert Data Engineer building an evaluation dataset for a RAG system.
    I will provide a raw text document. Your job is to generate {num_cases} test cases based ON THIS TEXT ALONE.
    
    RAW DOCUMENT:
    {raw_text}
    
    INSTRUCTIONS:
    1. Read the document carefully.
    2. Generate questions that vary in difficulty (some simple facts, some requiring reasoning across the text).
    3. The ground_truth must be 100% accurate according to the document.
    4. Output your response EXACTLY as a JSON object matching this schema:
    {json.dumps(schema_instructions, indent=2)}
    
    Return ONLY valid JSON.
    """
    
    print("Requesting synthetic test cases from the network...")
    
    try:
        response = await acompletion(
            model="gemini/gemini-2.5-flash",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3 # Slight creativity allowed here to generate diverse questions
        )
        
        raw_output = response.choices[0].message.content
        clean_string = raw_output.replace("```json", "").replace("```", "").strip()
        
        data_dict = json.loads(clean_string)
        validated_cases = TestCaseList(**data_dict)
        
        return validated_cases
        
    except Exception as e:
        print(f"Generation Pipeline Error: {e}")
        return None


if __name__ == "__main__":
    # A mock "document" from an imaginary corporate handbook
    mock_document = """
    Aegis Corp Employee Travel Policy (Effective 2026):
    Employees are allotted a $150 per diem for domestic food expenses, and $250 for international travel. 
    Flights over 6 hours may be booked in Premium Economy. Business class is strictly reserved for C-suite executives.
    Reimbursement requests must be submitted via the Concur portal within 14 days of returning. 
    Late submissions will incur a 20% penalty deduction from the total payout.
    """
    
    async def run_lab():
        results = await generate_test_cases_from_chunk(mock_document)
        if results:
            print("\n--- SYNTHETIC DATASET GENERATED ---")
            for i, case in enumerate(results.test_cases, 1):
                print(f"\n[Test Case {i} | Difficulty: {case.difficulty}]")
                print(f"Q: {case.question}")
                print(f"A: {case.ground_truth}")
                
    asyncio.run(run_lab())