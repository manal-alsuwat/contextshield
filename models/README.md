# Shipped models (no download or Hugging Face token needed)

| Directory | Model | Source | License | Size |
|---|---|---|---|---|
| `deberta-v3-small-prompt-injection-v2-onnx/` | ProtectAI DeBERTa-v3-small prompt-injection v2, fp32 ONNX export + tokenizer (+ `quant_check.json`: int8 rejected) | https://huggingface.co/protectai/deberta-v3-small-prompt-injection-v2 (base: microsoft/deberta-v3-small, MIT) | Apache-2.0 | 568 MB (`model.onnx`, git LFS) |
| `all-MiniLM-L6-v2/` | sentence-transformers/all-MiniLM-L6-v2 (safetensors) | https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 | Apache-2.0 | 88 MB |
| `en_core_web_sm/` | spaCy `en_core_web_sm` 3.8.0 | https://github.com/explosion/spacy-models | MIT | 15 MB |

The shield loads these local copies first and sets `HF_HUB_OFFLINE=1` by default. Not shipped (not needed): the base DeBERTa
model (rejected by the latency gate) and the int8 quantized files (rejected by the recall check) — they are ignored by git.
