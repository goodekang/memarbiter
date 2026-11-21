# MemArbiter

Code for the paper **MemArbiter: Provenance-aware arbitration of conflicting and poisoned memories for trustworthy long-horizon LLM agents** (*Information Fusion*).

MemArbiter is a governance layer for long-term agent memory: it adjudicates records at write time, scores fitness-for-use at read time, and localises and repairs poisoned records after a verifier flag. PERSIST is the accompanying benchmark of memory conflicts and cross-session poisoning.

## Setup

```bash
pip install -r requirements.txt   # Python 3.10+
```

Set the API keys listed in `configs/backbones.yaml` (e.g. `OPENAI_API_KEY`).

## Data

PERSIST is built from [LongMemEval](https://github.com/xiaowu0162/LongMemEval) and [AgentDojo](https://github.com/ethz-spylab/agentdojo), which are not redistributed here. Place local copies under `data/raw/` (see `configs/persist.yaml` and `configs/train_instruction_clf.yaml`).

## Usage

```bash
# classifiers
python scripts/train_conflict_classifier.py --build-pairs
python scripts/train_instruction_classifier.py --build-pairs

# benchmark
python scripts/build_persist.py --split all

# evaluation
python scripts/evaluate.py --suite conflict --system memarbiter
python scripts/evaluate.py --suite poison --system memarbiter
python scripts/evaluate.py --suite ablation --ablation no_scorer
```

Hyperparameters are in `configs/` and prompts in `src/prompts.py`. Ablation variants are listed in `configs/ablations.yaml`.

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
