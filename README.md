# Position Coverage in Multi-Image Vision-Language Models

Code for *Position Coverage: Why Multi-Image Vision-Language Models Favor the First and Last Images*.

Multi-image vision-language models read nearly every image correctly but lose accuracy when a single forward pass
must combine per-image evidence into a set-level verdict ("Do all six images contain X?"). Off-the-shelf models
favor the first and last images, and after fine-tuning on two-image questions they largely ignore a missing object
in between. The repository contains the data builders, LoRA
training, first-token scoring, the reading-referenced decomposition, and every analysis behind the paper's tables
and figures.

## Layout

| Path | Content |
|---|---|
| `e0lib/` | Shared library: image and fact tables (`atoms_*`), set predicates and the ideal-integration reference (`e3.py`, `logic.py`), prompts, LoRA (`s1.py`), first-token scoring (`scoring.py`) |
| `configs/models.yaml` | Model entries (Hugging Face Hub ids) |
| `configs/sources.yaml` | CLEVR and COCO sources, target descriptors |
| `scripts/` | Data builders, training, evaluation, analyses (below) |
| `tests/` | Unit tests for the predicate logic and the null model |

## Setup

```bash
pip install -r requirements.txt
```

Gemma 3 needs `torch>=2.6` for its image attention mask; the other models also run with `torch 2.5`.
`transformers==5.2.0` was used for all results.

Data (place under `data/raw/` or set the environment variables):

- CLEVR v1.0 (`data/raw/CLEVR_v1.0.zip`)
- COCO 2017 train/val images and annotations (`data/raw/coco2017/`, or `COCO_ROOT`)
- Visual Haystacks, single-needle split (`data/raw/visual_haystacks/`, or `VHS_ROOT`)

Model notes:

- InternVL3.5-8B is evaluated with one 448x448 tile per image (`internvl35_8b_t1`), which keeps the visual-token
  budget comparable across families.
- Mistral-Small-3.2-24B: transformers needs the processor and tokenizer files of
  `Mistral-Small-3.1-24B-Instruct-2503`; place them next to the 3.2 weights and point `path` to that directory.
- Models split over several GPUs (`device_map: auto`) stage cross-GPU tensor copies through host memory
  (`e0lib/scoring.py`), which keeps results exact on machines with unreliable peer-to-peer copies.

## Pipeline

```bash
# facts and images
python scripts/build_atoms.py --source clevr
python scripts/build_atoms.py --source coco
python scripts/s1_build.py --extract          # two-image training items
python scripts/e3_build.py --extract          # evaluation grid (n = 2..6, all rotations) and 2+3-image training items
python scripts/p1_build.py                    # edge-decisive, 2+4-image and 1/8-dose designs
python scripts/sealed_build.py --extract      # held-out image groups

# training (LoRA, two seeds per design); run name = <name>_s<seed>
python scripts/s1_train.py --arm A --seed 1 --device cuda:0                                   # 2-img
python scripts/s1_train.py --arm A --name A23 --items data/e3/train_23.jsonl --seed 1 --device cuda:0
python scripts/s1_train.py --arm A --name A3edge --items data/p1/train_3edge.jsonl --seed 1 --device cuda:0
python scripts/s1_train.py --arm A --name A24 --items data/p1/train_24.jsonl --seed 1 --device cuda:0
python scripts/s1_train.py --arm A --name A23q --items data/p1/train_23q.jsonl --seed 1 --device cuda:0
#   other families: --model internvl35_8b_t1 --name ivA ...   |   --model gemma3_12b --name gmA ...
#   stated designs: s1_teacher.py / e3_teacher.py, then --arm D or --arm Dself

# evaluation (first-token Yes/No scores)
python scripts/s1_eval.py --run A23_s1 --data e3 --modes logit --device cuda:0
python scripts/s1_eval.py --run A23_s1 --data sealed_e3 --modes logit --device cuda:0
python scripts/s1_eval.py --run base --data e3 --modes logit gen --device cuda:0
```

## Paper results

| Result | Builder / evaluation | Analysis |
|---|---|---|
| Decomposition: in-context readings, text facts, aggregation loss | `dx_e3_build.py`, `s1_eval.py --data e3dx` | `dx_e3_analyze.py`, `fig_cells.py` |
| Position map, three families | `s1_eval.py --data e3` | `p2_analyze.py`, `fig_slots.py` |
| Real example and all image groups | -- | `fig_example.py` |
| Coverage vs. image count, dose, answer bias | `p1_build.py` | `p1_analyze.py`, `p1_auroc.py`, `p1_posthoc.py`, `fig_cells.py` |
| Held-out confirmation | `sealed_build.py`, `s1_eval.py --data sealed_e3` | `p4_analyze.py` |
| Probes and attention | `p3_extract.py` | `p3_analyze.py` |
| Stated-fact supervision, untrained predicates | `dx_force_eval.py`, `dx_ops_build.py` | `e3_analyze.py`, `e3_posthoc.py`, `dx_force_analyze.py`, `dx_ops_analyze.py` |
| Larger zero-shot models | `p5_e3subset_build.py`, `s1_eval.py --data p5e3` | `p5_analyze.py` |
| Visual Haystacks | `p5_vhs_build.py`, `s1_eval.py --data p5vhs` | `p5_analyze.py`, `p5b_posthoc.py` |

Outputs are written to `outputs/` (per run under `outputs/s1/runs/<run>/`). Confidence intervals use a cluster
bootstrap over image groups.

## Tests

```bash
python -m pytest tests
```
