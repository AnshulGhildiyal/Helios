import json
from pydantic import BaseModel, Field, ValidationError

class FaithfulnessMetric(BaseModel):
    reasoning: str = Field(
        description="Step-by-step logic explaining if the answer contradicts the context."
    )
    score: int = Field(
        description="Score from 1 to 5. 5 means perfectly faithful to the context.", 
        ge=1, le=5
    )
    hallucination_flag: bool = Field(
        description="True if the answer contains details not found in the context."
    )

def generate_eval_prompt(question: str, context: str, answer: str) -> str:
    """Constructs a strict prompt forcing a JSON response."""
    
    schema_instructions = FaithfulnessMetric.model_json_schema()
    
    prompt = f"""
    You are an impartial, expert AI metric evaluator. 
    Your task is to measure the 'Faithfulness' of an Answer given its Context.
    
    QUESTION: {question}
    CONTEXT: {context}
    ANSWER: {answer}
    
    INSTRUCTIONS:
    1. Analyze the Answer against the Context.
    2. Does the Answer invent any new numbers, facts, or entities? If yes, hallucination_flag is true.
    3. Output your response EXACTLY as a JSON object matching this schema:
    {json.dumps(schema_instructions, indent=2)}
    
    Return ONLY valid JSON. No markdown formatting, no conversational filler.
    """
    return prompt

def parse_llm_response(raw_llm_string: str) -> FaithfulnessMetric:
    try:
        clean_string = raw_llm_string.replace("```json", "").replace("```", "").strip()
        data_dict = json.loads(clean_string)
        validated_metric = FaithfulnessMetric(**data_dict)
        
        return validated_metric
    
    except (json.JSONDecodeError, ValidationError) as e:
        print(f"CRITICAL ERROR: LLM broke the schema contract. Details: {e}")
        return None

if __name__ == "__main__":
    mock_llm_output = '{"reasoning": "The context says revenue was $5M, but the answer said $50M. This is a direct contradiction.", "score": 1, "hallucination_flag": true}'
    
    print("Testing the Metric Parser...\n")
    result = parse_llm_response(mock_llm_output)
    
    if result:
        print(f"Success! Parsed Object:")
        print(f"Score: {result.score}/5")
        print(f"Hallucination Detected: {result.hallucination_flag}")
        print(f"Reasoning: {result.reasoning}")