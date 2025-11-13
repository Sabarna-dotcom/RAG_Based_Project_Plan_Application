
# **README.md**

This project implements a complete solution for the Zoma AI Engineer assignment. The goal is to build a working pipeline that identifies the top three similar past opportunities, generates a structured phase-based project plan using an LLM, and validates the final output against a strict JSON schema.

Everything is deterministic, modular, and designed to run offline except for the LLM call. The plan is generated through Groq’s API using LangChain.

---

## **Project Structure**

### Similarity Search

The similarity engine is implemented in `similarity.py`.
It combines TF-IDF with dense embeddings from a local Ollama model and uses FAISS for fast vector search.
The engine prepares text features, generates embeddings, builds the FAISS index, and returns the three most relevant opportunities using a weighted hybrid scoring system.
Joblib is used to cache embeddings and TF-IDF matrices so repeated runs do not recompute them.

### Plan Generation

The generation logic lives in `generator.py`.
LangChain and ChatGroq are used to produce a realistic project plan that matches the assignment schema exactly.
A strict prompt forces the model to output only JSON and prevents schema echoing or extra fields.
Automatic retries are built in to correct invalid outputs, and the final result is stored in `plan.json`.

### Validation

The assignment’s original `validator.py` validates the generated JSON and confirms it follows the schema precisely.
It checks required fields, phase IDs, duration limits, dependencies, and disallows any additional properties.

---

## **Pipeline Flow**

`main.py` orchestrates the entire system
It loads the data, retrieves similar opportunities, calls the LangChain generator, writes the output to `plan.json`, and runs the official validator.
Running `python main.py` produces a fully valid project plan.

---

## **Design Decisions**

A hybrid similarity approach was chosen because technical project descriptions benefit from both lexical and semantic matching.
FAISS provides efficient similarity search even as the dataset grows.
LangChain gives a clean way to define prompts and parse JSON.
Groq models offer fast, low-latency inference suitable for repeated retries.
The official validator script from the assignment is used directly to prevent schema drift.

---

## **Guardrails and Reliability**

The LLM output is forced into strict JSON mode.
Only schema-compliant keys are allowed, and retries occur automatically when needed.
The pipeline halts if validation fails.
Embedding and TF-IDF caching ensures repeatable and stable similarity scoring.

---

## **Scaling Considerations**

FAISS can scale to thousands of opportunities with minimal performance impact.
Embeddings can be precomputed or generated asynchronously.
More metadata filters could be included later, such as industry or geography.
LLM function-calling can replace prompting for additional robustness.

---

## **Next Steps**

Possible enhancements include incremental FAISS updates, additional ranking signals from past project metadata, fallback LLM configurations, and an interactive UI for plan review.

---

