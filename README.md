# Feasibility Limits of Matched-Foil Controls for Doctored-Passage Citation Probes

Code, data and results for this paper.

## Layout

| Path | What it holds |
|---|---|
| `paper/` | The manuscript, its LaTeX source and its bibliography. (6 files, 364.6 KB) |
| `code/` | The experiment code. (47 files, 433.5 KB) |
| `results/` | The per-seed measurements and the results table. (4 files, 378.1 KB) |
| `figures/` | The figures and graphics in the paper, with the script that draws them. (20 files, 1.3 MB) |

## The experiment

Conditions measured: `baseline_unedited_alce_citation`, `self_span_end_injection`, `topical_foil_end_injection`, `entity_swap_foil_end_injection`, `self_span_random_injection`, `topical_foil_random_injection`, `null_random_drop_trailing_whitespace_edit`, `positive_control_planted_evidence_relocated`.

Seeds: 0, 1, 2, 3. A separate confirmation run used 101, 102, 103, held back from every earlier run.

The paper's headline numbers come from that confirmation run.

### Limitations of this run

- The run ended early, after recording 4 seeds.
- The study is a feasibility study, and every result in it is exploratory, since the yield analysis was specified after both runs had finished.
- The probe admitted 2 of 404 items on the reserved pool and 4 of 544 on the development pool, far short of the design targets.
- Every reserved-pool seed failed the positive-control validity check, so the per-arm migration counts are diagnostics and support no comparison between conditions.

## Data

`code/data.py` downloads the datasets on first run, into the directory `RC_DATA_ROOT` names.

## Running the experiment

```bash
python -m venv .venv && .venv/Scripts/activate      # Windows
pip install "torch>=2.9.0" --index-url https://download.pytorch.org/whl/cu128   # CUDA build
pip install -r code/requirements.txt
set RC_DATA_ROOT=%CD%\data
python code/setup.py                                # prepares the corpus
python code/main.py
```

A run writes its measurements in the same shape as `results/measurements.json`.

