
# Zoma AI Engineer — Take‑Home Assignment (2–3 hours)

**Objective**  
Build a small prototype that:  
1) finds the **top‑3 similar past opportunities** for a new opportunity,  
2) generates a **project phase breakdown** in **strict JSON** conforming to `output_schema.json`, and  
3) explains your design choices and trade‑offs.

---

## Provided Files
- `dataset.json` — 12 real‑looking past opportunities (title, description, components, tags, effort).  
- `new_opportunity.json` — the new opportunity to plan.  
- `output_schema.json` — strict schema for your generated plan.  
- `validator.py` — quick script to validate your output JSON against the schema.

---

## What You Need to Deliver
- **Code** (Python or Node.js) that:
  - Loads `dataset.json` and `new_opportunity.json`.
  - Computes top‑3 similar opportunities (embeddings, TF‑IDF, or hybrid — your choice).
  - Prompts an LLM **or** produces a **mock LLM** (deterministic function) to generate a plan matching `output_schema.json`.
  - Validates the output using JSON Schema and **fails fast** on mismatch.
- **Output file** named `plan.json`.
- **README.md** (≤1 page) with:
  - *Approach & Architecture* (similarity strategy, generation, validation).
  - *Guardrails* (how you keep JSON valid; retries, temperature, constrained decoding, function‑calling, etc.).
  - *Scaling Considerations* (latency, cost, vector updates, metadata filters).
  - *Next Steps* (what you’d productionize first at Zoma).

> **Timebox:** Aim for **2–3 hours**. It’s okay to implement a simplified version — just document trade‑offs.

---

## Functional Requirements
1. **Similarity Search**
   - Combine at least **title + description + tags** for text similarity.
   - Return exactly **3 items** with their `id`, `title`, and a **relevance score** (`0.0–1.0`).
   - Deterministic behavior: set a seed / fixed parameters.

2. **Generation**
   - Create between **3–8 phases**, each with an `id` like `PH-001`, `PH-002`, …
   - Enforce **strict JSON** of `output_schema.json`.
   - Use the similar items to calibrate phase durations and add a short `notes` field explaining rationale.

3. **Validation**
   - Validate `plan.json` against `output_schema.json` (use the provided `validator.py` or your own).  
   - If invalid, **retry** up to 2 times with a different strategy (e.g., add a system reminder, reduce temperature, or drop free‑text).

4. **CLI**
   - Provide a `run.sh` or `npm run start` that produces `plan.json` in the project root.

---

## Non‑Functional Requirements
- **Code Quality:** small, readable modules; clear function names.
- **Reproducibility:** deterministic results without internet.
- **Performance:** keep end‑to‑end under a few seconds on a laptop (mocked LLM is fine).
- **Security:** do not commit secrets; no network calls required.

---

## Allowed Tools
- Python (suggested libs: `scikit-learn`, `numpy`, `jsonschema`) or Node.js (e.g., `ajv`, any local text similarity lib).
- You may mock the LLM. If you do use an API, gate it behind a flag and provide a mocked default.

---

## Acceptance Criteria (What we check)
- Output matches **schema** and is **deterministic**.
- Top‑3 similar opportunities look **reasonable** given the new opportunity.
- Clear **guardrails** to produce valid JSON (no trailing commas, no prose).
- Code is easy to read and run via a single command.
- README demonstrates practical **production thinking** for Zoma.

---

## Bonus (Optional, pick any)
- **Hybrid search**: blend TF‑IDF cosine with a simple embedding (e.g., MiniLM) if available offline.  
- **Metadata filters**: prefer same industry/country when scores tie.  
- **Cost awareness**: show a token/latency budget if you call an LLM.  
- **Unit test** for the validation/retry loop.

---

## Submission
- Send a ZIP or GitHub repo link with instructions.  
- Include a short note on how you’d integrate this into Zoma’s existing **Bottom‑Up**/**Top‑Down** AI flows.

**Good luck!** — Zoma Team
