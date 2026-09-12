# MemArbiter

Code and configuration for the paper:

> **MemArbiter: Provenance-aware arbitration of conflicting and poisoned memories for trustworthy long-horizon LLM agents**  
> *Information Fusion*

MemArbiter is a governance layer for long-term agent memory: it adjudicates records at write time, scores fitness-for-use at read time, and localises / repairs poisoned entries after a verifier flag. PERSIST is the accompanying 3,100-instance benchmark (conflicts + cross-session poisoning).

## Setup

```bash
pip install -r requirements.txt
```

Python 3.10+ is required. Set the API keys used by `configs/backbones.yaml` (e.g. `OPENAI_API_KEY`).

## Data

PERSIST is derived from [LongMemEval](https://github.com/xiaowu0162/LongMemEval) and [AgentDojo](https://github.com/ethz-spylab/agentdojo). Those corpora are not redistributed here. Point `configs/persist.yaml` at local dumps under `data/raw/`.

## Usage

```bash
python scripts/train_conflict_classifier.py --build-pairs
python scripts/train_instruction_classifier.py --build-pairs
python scripts/build_persist.py --split all
python scripts/evaluate.py --suite conflict --system memarbiter
```

Hyperparameters match the paper (Appendix A.2) and live in `configs/`. Prompts are in `src/prompts.py`.

## Citation

```bibtex
@article{memarbiter2026,
  title={MemArbiter: Provenance-aware arbitration of conflicting and poisoned memories for trustworthy long-horizon {LLM} agents},
  journal={Information Fusion},
  year={2026}
}
```

## License

MIT
