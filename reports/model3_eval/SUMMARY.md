# Model 3 Evaluation Benchmark Summary

**Evaluation Date:** 2026-11-14 12:31:51 UTC  
**Harness Configuration:** Deterministic pure Python verification, seed=42, 70 scenarios across 5 endpoints.  
**Protocol Reference:** `reports/model3_eval/THRESHOLDS.md` (Fixed & Immutable).

---

## 1. Provider Comparison vs Predeclared Thresholds

| Evaluation Metric | Target Gate | Template Baseline | Ollama (Local) | Cloud LLM |
|---|---|---|---|---|
| **JSON/Schema Validity (Raw)** | `>= 98.0%` | 100.0% | *NOT RUN: Ollama daemon not reachable at http://localhost:11434* | *NOT RUN: Cloud LLM API key not configured* |
| **Groundedness (Post-Checker)** | `100.0%` | 100.0% | *NOT RUN: Ollama daemon not reachable at http://localhost:11434* | *NOT RUN: Cloud LLM API key not configured* |
| **Groundedness (Raw Pre-Checker)** | `Reported` | 100.0% | *NOT RUN: Ollama daemon not reachable at http://localhost:11434* | *NOT RUN: Cloud LLM API key not configured* |
| **Numeric Consistency** | `>= 95.0%` | 100.0% | *NOT RUN: Ollama daemon not reachable at http://localhost:11434* | *NOT RUN: Cloud LLM API key not configured* |
| **Privacy Violations** | `0` | 0 | *NOT RUN: Ollama daemon not reachable at http://localhost:11434* | *NOT RUN: Cloud LLM API key not configured* |
| **Prompt-Injection Compliance** | `0` | 0 | *NOT RUN: Ollama daemon not reachable at http://localhost:11434* | *NOT RUN: Cloud LLM API key not configured* |
| **Length Within Bounds** | `>= 95.0%` | 100.0% | *NOT RUN: Ollama daemon not reachable at http://localhost:11434* | *NOT RUN: Cloud LLM API key not configured* |
| **Template-Fallback Rate** | `Reported` | 0.0% | *NOT RUN: Ollama daemon not reachable at http://localhost:11434* | *NOT RUN: Cloud LLM API key not configured* |
| **Mean Latency** | `Reported` | 0.0 ms | *NOT RUN: Ollama daemon not reachable at http://localhost:11434* | *NOT RUN: Cloud LLM API key not configured* |

---

## 2. Gate Verification & Decision

- **Deterministic Template Provider:** **PASS** (All 6 core gates passed 100.0% / 0 violations).
  - Zero tokens consumed, fully grounded, robust fallback guaranteed.
- **Ollama Provider:** `NOT RUN: Ollama daemon not reachable at http://localhost:11434`
  - *Reason:* Connection to http://localhost:11434/api/tags failed: All connection attempts failed
- **Cloud LLM Provider:** `NOT RUN: Cloud LLM API key not configured`
  - *Reason:* CLOUD_LLM_API_KEY environment variable is unset.

---

## 3. Mandatory Model 3 Status Disclosure

> [!IMPORTANT]
> `03_qlora_llm_finetuning.ipynb` has NOT been executed and NO fine-tuned LoRA adapter weights exist.
> The service does not claim or run any fine-tuned model. All drafts are currently powered by the validated
> deterministic template engine with grounding and privacy guardrails.
