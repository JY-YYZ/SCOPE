# SCOPE (KDD 2026)

- Paper DOI: 10.1145/3770855.3817993
- Author: Xi'an Jiyun Technology Co., Ltd.


# SCOPE Core Release

This repository contains a compact reference implementation of the core SCOPE-style editing pipeline. It keeps only the method components needed to build an activation covariance shield, compute a null-space projection, estimate a refusal direction, and apply projected model updates.

## Files

- `run_scope_edit.py`: End-to-end example runner. It loads a causal language model, builds projection matrices from general/expert texts, extracts refusal directions from prompt/refusal pairs, and saves an edited model.
- `scope/covariance.py`: Online activation covariance estimator using a Woodbury inverse update. It also builds the null-space projection used to preserve high-energy activation subspaces.
- `scope/directions.py`: Utilities for resolving model modules and extracting a refusal direction from harmful-prompt/refusal-anchor activation differences.
- `scope/editing.py`: Gradient projection and refusal-alignment loss used during editing.
- `scope/data.py`: Small text/JSONL loaders plus tiny toy fallback samples for smoke tests.
- `requirements.txt`: Minimal Python dependencies.

## Data Format

General and expert data are plain text files with one sample per line.

Safety data is JSONL. Each row should contain a harmful prompt and a refusal anchor:

```json
{"prompt": "harmful instruction here", "refusal": "I cannot help with that request."}
```

The loader also accepts common aliases such as `harmful_prompt`, `question`, `safe_response`, and `answer`.

## Example

```bash
pip install -r requirements.txt

python run_scope_edit.py \
  --model /path/to/base-model \
  --output /path/to/edited-model \
  --target-modules model.layers.15.mlp.down_proj \
  --general-texts data/general.txt \
  --expert-texts data/expert.txt \
  --safety-pairs data/safety_pairs.jsonl \
  --max-cov-batches 32 \
  --edit-steps 100 \
  --kl-weight 0.05
```

The default target module is only an example for Llama-like architectures. For other models, pass the module names you want to edit with `--target-modules`.

## Notes

This is a core-code release rather than a full experiment reproduction package. The fallback samples in `scope/data.py` are only for checking that the script runs; meaningful editing requires real general-domain, expert-domain, and safety-pair data.
