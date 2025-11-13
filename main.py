import json
import sys
import os

from similarity import HybridSimilarity
from generator import generate_plan
from validator import main as validator_main


def main():
    print("Pipeline Started")

    DATASET_PATH = "dataset.json"
    NEW_OP_PATH = "new_opportunity.json"
    SCHEMA_PATH = "output_schema.json"
    PLAN_PATH = "plan.json"


    # Run Hybrid Similarity Search
    print("\nComputing top-3 similar opportunities...\n")

    sim_engine = HybridSimilarity()
    sim_engine.fit(DATASET_PATH)
    similar_items = sim_engine.query(NEW_OP_PATH)

    print("Top-3 similar opportunities:")
    print(json.dumps(similar_items, indent=2))

    # Generate plan.json using LLM
    print("\nGenerating plan.json using LLM...\n")

    result = generate_plan(
        similar_items=similar_items,
        new_opportunity_path=NEW_OP_PATH,
        schema_path=SCHEMA_PATH,
        output_path=PLAN_PATH
    )

    # Validate using official validator.py
    print("\nValidating plan.json with validator.py...\n")

    code = validator_main(PLAN_PATH, SCHEMA_PATH)

    if code == 0:
        print("\nplan.json is fully valid and ready!")
    else:
        print("\nplan.json failed validation.")
        sys.exit(1)

    print("\nDone. Output saved to:", PLAN_PATH)


if __name__ == "__main__":
    main()
