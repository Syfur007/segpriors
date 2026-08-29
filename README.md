# segpriors

**A config-driven PyTorch framework for a single-encoder input-representation study in medical
image segmentation.**

> One YAML fully determines a run. Models are trained, evaluated, and statistically validated
> under one shared pipeline — training, evaluation, statistics, attribution, robustness, and
> reporting are each a first-class module, not a notebook.

## 1 · QUICKSTART

| Step | Command |
| --- | --- |
| Install | `conda activate segpriors && pip install -r requirements.txt` |
| Test | `pytest tests/ -v` |
| Train | `python train.py --config configs/experiment/mkunet/mkunet_t_clinicdb.yaml` |
| Evaluate | `python eval.py --config <same config> --fold 0 --allow-test-eval` |
| Reproduce | `./scripts/reproduce.sh` — a real, reduced end-to-end pass |

## 2 · LAYOUT

| Layer | Package(s) | Purpose |
| --- | --- | --- |
| Config | `configs/`, `orchestration/schema.py` | `compose:`-merged YAML, schema-validated on load |
| Data | `datasets/` | Loaders, channel construction, augmentation, leakage guards |
| Models | `models/` | UNet family — one registry |
| Training | `training/`, `losses/` | Trainer, optimizer, determinism, declarative losses |
| Metrics | `metrics/` | The one canonical Dice / IoU / HD95 / ASD / NSD / ECE source |
| Orchestration | `orchestration/` | Run manifest, ledger, sweeps |
| Analysis | `stats/`, `profiling/`, `attribution/`, `robustness/`, `analysis/` | Significance testing, efficiency, explainability, robustness |
| Reporting | `reporting/` | Manuscript tables/figures, blocking rules |

## 3 · MODELS

| Registry name | Kind |
| --- | --- |
| `unet`, `attention_unet` | Baselines |
| `mk_unet` (+ `_s` / `_t`), `emcad` | Baselines |

`mk_unet` is the primary architecture under study; `unet` is used as a generality check. Every
model here is single-encoder.

## 4 · CHANNEL MODES

| Mode | Groups |
| --- | --- |
| m1 | RGB |
| m2 | RGB + XY |
| m3 | RGB + YCbCr |
| m4 | RGB + XY + Rθ |
| m5 | RGB + XY + YCbCr + Rθ |

## 5 · GUARANTEES

| Guarantee | Enforced by |
| --- | --- |
| Test set touched once, on purpose | `datasets.datamodule.get_test_loader(token)` raises without a minted token |
| Every run is addressable | `run_id = R-{config_hash[:7]}-s{seed}-f{fold}`, in every manifest/checkpoint/artefact |
| No config-drift between models | Shared `AugmentationPolicy`; no per-model augmentation key in the schema |
| Reported tables are trustworthy | `reporting/` refuses a dirty-tree run, an under-seeded config, an unstated comparison, or unsanitised saliency |
