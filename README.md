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

The quarantine-gate classifier additionally reads (paths in `configs/train_instruction_clf.yaml`):

| Path | Content |
|---|---|
| installed `agentdojo` package | user tasks, injection tasks and seed data of the four environments |
| `data/raw/agentdojo_runs/` | AgentDojo trace logs (`--logdir`) of benign runs; only runs without an injection task are read |
| `data/raw/asb/all_attack_tools.jsonl` | local copy of the ASB attack-tool list |
| `data/raw/gate_benign_imperative.jsonl` | realistic benign texts that address the reader (`text`, `category` in `email_message` / `web_page` / `documentation`); test partition only |

## Usage

```bash
python scripts/train_conflict_classifier.py --build-pairs
python scripts/train_instruction_classifier.py --build-pairs          # grouped split (Table A.1)
python scripts/build_persist.py --split all
python scripts/evaluate.py --suite conflict --system memarbiter
```

Hyperparameters match the paper (Table A.3) and live in `configs/`. Prompts are in `src/prompts.py`.

### Quarantine-gate classifier (Appendix A.1)

`--build-pairs` collects benign tool outputs from executions of the AgentDojo user tasks (trace logs of benign runs and ground-truth executions of every user task, with the default values of all injection vectors), adds emails, calendar entries, channel messages, web pages and files from the environment seed data, and rewrites the AgentDojo injection tasks and ASB attack tools in the three styles and several carriers (`src/train/gate_data.py`). Every row records its group.

- `--split grouped` (default): injections are grouped by attack specification, benign tool returns by user task and seed items by template family; train/dev/test are formed by group, texts with cosine similarity above 0.95 to an earlier partition are removed, the threshold is chosen for 98% recall on dev, and precision / recall / FPR are reported on the held-out test partition, overall and per category, together with a random row-level split of the same rows (`checkpoints/instruction_clf/heldout_report.json`, split written to `data/classifier/instruction_split.jsonl`).
- `--split row`: the original random row-level split (`checkpoints/instruction_clf_rowsplit/`). The gate used for the main-table results was trained this way; point `instruction_classifier.checkpoint` / `threshold_file` in `configs/memarbiter.yaml` at that directory to reproduce them.

### Update rules: revised method vs. main-table configuration

`configs/memarbiter.yaml` defaults to the revised rules:

- `corroboration: restricted` -- a `CONSISTENT` pair corroborates only if both records describe the same entity and attribute (backbone `same_attribute` field for escalated pairs, entity--slot tags otherwise) and their provenance is independent (different sources, no derivation path or common derivation ancestor); a source contributes at most one corroboration per attribute.
- `lineage: typed` -- derivation and supersession edges are stored separately; taint inheritance, the attribution candidate set and quarantine follow derivation edges only, and a record that superseded the responsible record is retained.

The main tables were produced with the original rules; `--ablation original_rules` (unrestricted corroboration, single untyped edge set) reproduces them, and `unrestricted_corroboration` / `untyped_lineage` switch one rule at a time (Table A.2, Section 7.12).

### Query context psi

`--query-context annotated` (default) supplies the scope annotation of each PERSIST-Conflict query as psi, as in the main tables. `--query-context extracted` derives psi from the query and the preceding turns of the session by lexical matching against the condition keys of the stored scope predicates, without an LLM call (`src/memarbiter/context.py`; Table 14).

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
