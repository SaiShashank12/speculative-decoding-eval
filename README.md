# Fast Inference from Transformers via Speculative Decoding

Implementation and evaluation of speculative decoding techniques for accelerating Large Language Model inference.

## Overview

This project compares three decoding strategies:
1. **Vanilla LLM** — Standard autoregressive generation
2. **Vanilla Speculative Decoding** — Using a separate draft model for candidate generation
3. **EAGLE-3** — Feature-level speculative decoding

## Project Structure

```
├── run_benchmark.py          # Main benchmarking script
├── EAGLE/                    # EAGLE framework (patched for MPS compatibility)
├── overleaf_update2/         # Project Update 2 (ACL LaTeX)
│   ├── main.tex
│   ├── custom.bib
│   ├── acl.sty
│   └── acl_natbib.bst
├── proposal.pdf              # Original project proposal
└── README.md
```

## Quick Start

```bash
# Create virtual environment
python3 -m venv ea_env
source ea_env/bin/activate

# Install dependencies
pip install -r EAGLE/requirements.txt

# Run benchmark
python run_benchmark.py
```

## Current Results (Apple Silicon MPS)

| Method | Tokens/sec | Relative |
|--------|-----------|----------|
| Vanilla LLM | 44.17 | 1.00× |
| Vanilla Speculative | 30.38 | 0.69× |
| EAGLE-3 | 9.07 | 0.21× |

> **Key Finding:** Speculative decoding shows *slowdowns* on Apple Silicon due to the Unified Memory Architecture eliminating the memory-bandwidth bottleneck that speculative decoding exploits.

## Models Used

- **Target:** `Qwen/Qwen3-1.7B`
- **Draft:** `Qwen/Qwen2.5-0.5B`
- **EAGLE:** `AngelSlim/Qwen3-1.7B_eagle3`

## Author

**Sai Shashank Mudliar** — Purdue University (`smudliar@purdue.edu`)
