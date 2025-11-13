import json
import os
from typing import Dict, List
import time
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from validator import main as validator_main

load_dotenv()

# LangSmith Tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "RAG Based Project Plan Generator"


# Load Schema
def load_schema(schema_path="output_schema.json"):
    with open(schema_path, "r") as f:
        return json.load(f)


# Initialize LLM
def init_llm():
    return ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant",
        temperature=0.0,
        max_tokens=2048
    )


# Generator function
def generate_plan(similar_items, new_opportunity_path="new_opportunity.json", schema_path="output_schema.json", output_path="plan.json"):

    # Load input JSONs
    with open(new_opportunity_path, "r") as f:
        new_op = json.load(f)

    schema = load_schema(schema_path)

    new_op_str = json.dumps(new_op, indent=2)
    sim_str = json.dumps(similar_items, indent=2)
    schema_str = json.dumps(schema, indent=2)

    # Prompt Template
    prompt = PromptTemplate(
    input_variables=["new_op", "similar_items", "schema"],
    template="""
        You are a project planning expert. 
        Your task is to CREATE a project plan JSON that MATCHES the schema, NOT to output the schema itself.

        VERY IMPORTANT RULES:
        - DO NOT repeat or reproduce the schema.
        - DO NOT repeat or reproduce the opportunity input.
        - DO NOT add extra fields outside the schema.
        - DO NOT output comments, markdown, or explanations.
        - DO NOT output the schema itself.
        - You MUST generate a NEW plan with realistic values.

        Your output MUST:
        - Contain ONLY: opportunity_id, phases, total_duration_days, notes.
        - Generate 3–8 phases.
        - Phase IDs must be sequential: PH-001, PH-002, ...
        - duration_days must be integers 1–60.
        - total_duration_days = sum of all phase duration_days.
        - dependencies must reference earlier phase IDs only.

        REFERENCE INPUT — READ ONLY, DO NOT COPY:
        NEW OPPORTUNITY:
        {new_op}

        TOP-3 SIMILAR OPPORTUNITIES (use them to size durations):
        {similar_items}

        STRICT SCHEMA (FOLLOW THE STRUCTURE, NOT THE CONTENT):
        {schema}

        Now GENERATE a realistic project plan as a JSON object matching the schema.
        Do NOT output the schema. 
        Do NOT output explanations.
        Return ONLY the JSON object:
        """
    )


    llm = init_llm()
    parser = JsonOutputParser()

    # LCEL chain
    chain = prompt | llm | parser

    # Retry loop
    for attempt in range(1, 4):
        print(f"\nAttempt {attempt} to generate valid JSON...")

        try:
            result = chain.invoke({
                "new_op": new_op_str,
                "similar_items": sim_str,
                "schema": schema_str
            })

            # Save before validating
            with open(output_path, "w") as f:
                json.dump(result, f, indent=2)

            print("Running official validator.py...")
            code = validator_main(output_path, schema_path)

            if code == 0:
                print("plan.json is VALID")
                return result
            
            print("validator.py rejected JSON. Retrying...\n")

        except Exception as e:
            print(f"Error: {e}")
            print("Retrying...\n")

        time.sleep(1)

    raise RuntimeError("Failed to generate valid JSON after 3 attempts.")




# Manual test
if __name__ == "__main__":
    dummy_similar = [
        {"id": "A1", "title": "Mock A", "relevance_score": 0.92},
        {"id": "A2", "title": "Mock B", "relevance_score": 0.89},
        {"id": "A3", "title": "Mock C", "relevance_score": 0.85},
    ]

    generate_plan(dummy_similar)
