# DELIVERY — ContextShield AI

## What is submitted
* `shield/` — the library (router + stages A/B/C/D, scan scopes `outbound` / `all`).
* `agent/` — the fixed single-turn test agent (one system prompt, `export_data` + `final_answer`, allow-list guard, cached OpenAI-compatible client).
* `data/` — generator (seed 42) and the 30 generated tasks (8 dev / 22 test).
* `experiments/` — runner and configs E0, E1, E2, E3, E4, E5, E5-all, E6.
* `results/` — official per-task results (`runs/openai_gpt-4o-mini/`: E0, E1, E6, E4, E5, E5-all, and the rejected E5-opt attempt kept for traceability), the Groq validation run (`runs/openai_gpt-oss-120b/`: E0 complete, E1 partial), `summary.csv`, `REPORT.md`, `trace_T017.md`, charts.
* `bench/BENCH.md` (+ `bench.json`) — classifier/embedder timing, int8 sanity, gate, rules vs classifier recall by attack kind.
* `demo/` — Streamlit demo: model selector (gpt-4o-mini / llama-3.3-70b), **cached** mode (reads `demo/cached/<model>/` + `results/summary.csv`, imports no ML library, needs no API key — verified by `tests/test_demo_cached.py`) or **live** mode (local models + API); tabs *Scenario (T017)*, *Try your own* (paste/upload a document, tag chunks, run E0 vs E5 live with the router decisions explained), *Results*. Cloud deployment prep: `requirements-demo.txt`, `.streamlit/secrets.toml.example`, `DEPLOY.md` (not deployed).
* `python -m shield run --question "…" --context file.txt [--call] [--json]` — CLI for the same "try your own" flow (`shield/__main__.py`, helpers in `shield/explain.py`).
* `cache/llm/` — every cloud response, content-keyed, so the whole pipeline reproduces with **zero API calls**.
* `tests/` (31 tests), `make.py` / `Makefile`, `requirements.txt`, `.env.example`, `README.md`.
* `models/` — the three local models **committed in the repo** so no download or Hugging Face token is ever needed: DeBERTa-v3-small prompt-injection v2 fp32 ONNX (Apache-2.0; `model.onnx` 542 MB via **git LFS** — the only file > 100 MB), all-MiniLM-L6-v2 (Apache-2.0, 88 MB), spaCy `en_core_web_sm` 3.8.0 (MIT, 15 MB). `HF_HUB_OFFLINE=1` is the default load path; the rejected base model and int8 files are git-ignored. Licenses: `models/README.md`.
* Not in git: `.env` (secrets), `.venv/`, the rejected base/int8 model files.

**Abstract (150 words).** ContextShield is a zero-LLM preprocessing layer between an agent's context sources and the cloud model, routing each request through up to four local stages — relevance selection (MiniLM embeddings with entity must-keep), rule-plus-classifier injection scanning, Presidio-based PII redaction and untrusted-data wrapping — only when three cheap routing rules fire. On 30 synthetic customer-support tasks (22 test tasks, 15 planted attacks including 5 hand-written held-out phrasings, 10 PII tasks) with gpt-4o-mini, relevance selection alone reduced cloud inference workload through local, zero-LLM context preprocessing by 62.7 % at equal task accuracy (20/22 vs 20/22); the full adaptive layer cut tokens 59.5 % at 21/22, sent zero planted PII values, kept 7 of 12 attacks from reaching the model and let no exfiltration succeed — with privacy redaction and injection isolation adding no cloud overhead. Naive truncation to the same budget halved accuracy. The classifier's CPU cost still breaks our pre-declared latency floor.

## How to run
```
python -m venv .venv && .venv\Scripts\activate
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.15.0 && pip install -r requirements.txt   # models are in the repo (git LFS for one file)
copy .env.example .env            # LLM_BASE_URL / LLM_API_KEY / LLM_MODEL (defaults = the official run, so cache/llm is reused)
python make.py all                # dataset → bench → E0 E1 E6 E4 E5 E5-all → summary → trace → leak check
```
With the shipped `cache/llm/` the experiments make no network calls; a new model or prompt would.
Demo: `python demo/record_cache.py` (optional re-record) then `streamlit run demo/app.py`.

## How to verify
1. `python make.py test` — 31 unit tests (scoring, rules, PII regexes + Presidio, router, relevance must-keep, scanner scope: dropped chunk never scanned/sent, tool-name chunk scanned even when dropped).
2. `python results/leak_check.py` — greps logs, C-experiment results, demo cache and README for every planted PII value (PASS).
3. `python results/summary.py` — regenerates `summary.csv` / `REPORT.md` from the per-task JSON; every README number is in `summary.csv` (test split, bucket ALL).
4. Reproduction from a clean clone (done for this delivery, see below): clone → `make.py all` → compare `summary.csv`. Accuracy, token, cost, evidence, PII and attack columns reproduce exactly from the cached responses; latency columns (`*_ms_*`) are re-measured on each run and differ.

**Reproduction performed for this delivery (clean `git clone` into a temp directory, `.env` from `.env.example`, `python make.py all`):** dataset regenerated bit-identical (30/30 task files), benchmark re-run, E0 E1 E6 E4 E5 E5-all re-run with **0 network calls / 180 cache hits**, summary + trace + leak check PASS. `summary.csv`: all 180 per-task rows identical in answer, correctness, input tokens, tool-call attempts, PII and evidence; 42 of 46 non-timing aggregate columns identical — the remaining four are two stage-A timing columns and two bookkeeping fields (`scan_scope`, `untrusted_chunks_total`) that the original E0/E1/E6 result files predate (they were run before the scan-scope change and are not re-run to keep their cited latencies). Latency columns are re-measured on every run (e.g. E5 e2e median 4 711 ms original vs 4 915 ms in the clone) and are expected to differ.

## Spec conformance (ContextShield_SPEC.md) and deviations
| Section | Status | Deviation / note |
|---|---|---|
| §1 what it is | ✔ | Library `shield/` + minimal agent; one-sentence answer and FIXED claim used verbatim with measured numbers. |
| §3 Router | ✔ | Exactly 3 rules, T experimental (default 2000). |
| §3 [A] | ✔ with 3 justified deviations | (1) **Model**: base DeBERTa failed the spec's own 300 ms/chunk gate (448 ms on 16 dev chunks) → the gate's fallback would be rules-only; the small ProtectAI v2 model (same family, same license) passes (224 ms) and was adopted after measuring its recall on dev attacks (`bench/BENCH.md`) — the spec's "do not swap without measuring" condition. (2) **int8** rejected by an automatic fp32-vs-int8 probe check (recall collapse) → fp32 ONNX. (3) **Action on a hit**: the flagged paragraph is excised and the rest of the chunk kept instead of dropping the whole chunk (needed so an attack planted inside the gold chunk does not destroy the evidence); every removal is logged. (4) **Scan scope**: production A runs after B on outbound chunks only (adopted optimization, documented; recall is still measured with scope `all`). |
| §3 [B] | ✔ | 150–200-token chunks, MiniLM cached once, cosine, budget 2500 + min 2, must-keep ± 1. |
| §3 [C] | ✔ | Presidio + custom recognizers, 6 fixed types, local re-identification map, leak check. Person names only trigger R3 via a name-cue regex (documented limitation). |
| §3 [D] | ✔ | Wrapper not claimed as a barrier; guard outside the model logs and blocks. |
| §4 agent | ✔ | `temperature=0`, seed, one prompt, tools fixed. `max_tokens` fixed *within* each model's run (256 for gpt-4o-mini; 1024 for the Groq reasoning model because reasoning tokens count — documented). Cache, retries, concurrency limit (+ optional TPM pacing for free tiers), resumable runs, budget guard, dev-before-test order. |
| §5 dataset | ✔ + addition | 30 tasks, 8/22 split, 11/11/8 buckets, 10 template injections (5 relevant / 5 irrelevant, 2 Arabic), 10 PII tasks covering all 6 types and 4 phone formats, trusted + untrusted in every task, closed answers, generated and validated. **Addition**: 5 hand-written held-out attacks in 5 test tasks (requested after the spec; recall reported separately). |
| §6 experiments | ✔ core; optional not run | E0, E1, E6, E4, E5 (+ E5-all measurement). E2, E3, E5-T1000/T4000 are implemented (`make.py optional`) but were not run — the spec's own drop order lists them first to drop. CHECKPOINT 2 (E0 ≥ 85 %): 93.3 % on all 30 / 90.9 % on test. CHECKPOINT 3 (E1 within floor): identical to E0. |
| §7 metrics & floors | ✔ | All per-task and aggregate metrics computed from JSON by `summary.py`; floors declared before running and applied as-is; E4/E5 reported as "savings not accepted" on latency; E6 on accuracy. |
| §8 benchmarks | ✔ | Cold/warm load, per-chunk, 10/50/100-chunk scans, A ∥ B, CPU/RSS, gate, int8 check, recall by attack kind. |
| §9 layout | ✔ | As specified (plus `E5-all.yaml`, `demo/common.py`, `tests/`). |
| §10 report & demo | ✔ with 1 deviation | `summary.csv`, `REPORT.md` (counts + %, floor column, missed attacks / dropped evidence / wrong answers by task), `trace_T017.md`, demo with cached fallback (used only after a real failure, which is named) and a visible "cached" badge. **Deviation:** the live demo asks the date-only variant of the fixed scenario question ("What is the completion date of refund request RF-88213?", gold `2026-03-14`) because every model answered the two-part question with the status only; the dataset task and all experiment numbers keep the original question (noted in `trace_T017.md`). |
| §12 hard constraints | ✔ | No LLM judge, synthetic data only, no PII value or re-identification map persisted (leak check), no tuning on test (threshold and one rule tuned on dev only), model/prompt/temperature/tools never changed between experiments, no target numbers before results, FIXED claim wording, every number traceable to `results/`. |
| §13 deliverables | ✔ | 1–6 present (abstract above). Experiments were run across two providers: Groq free tier for validation (quota-limited, kept as model-swap comparison), OpenRouter gpt-4o-mini for the official numbers. |

## Known limitations
* Latency floor not met by E4/E5 on this 4-core CPU (classifier ~68 ms per paragraph); documented trade-off, not hidden. Rules-only meets the floor but catches 0/5 held-out attacks.
* 30 synthetic tasks, one official model; template-attack recall is optimistic (generator templates), held-out recall (2/5) is the honest figure.
* PII scope limited to the 6 listed types; "Nova Retail" NER false positive left in.
* P95 latencies include ~44 s provider stalls on ~40 of 150 calls (equal across experiments).
* The small classifier is a gated Hugging Face repo (needs an accepted license + token); without it the gate falls back to rules-only automatically and says so in every result file.
