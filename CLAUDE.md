# CLAUDE.md

Notes for future Claude sessions in this repo. Keep this short; treat the README
as the human-facing source of truth and this file as the working memory for an
AI collaborator.

## What this project is

A NARX (Nonlinear AutoRegressive with eXogenous inputs) modelling pipeline for
dynamical systems, implemented in PyTorch. Trains either a feed-forward MLP
(`rnn: false`) or a multi-layer ReLU RNN (`rnn: true`) on experimental
input/output CSVs, then runs closed-loop predictions on held-out data and
user-defined test vectors. Originally built around a magnetic brake (RM42), but
the pipeline is dataset-agnostic — only `utils/data_processing.py` is
brake-specific.

## Layout cheat sheet

```
main.py                   Interactive menu: preprocess / train / predict / clean
config.py                 Module-level globals; populated by load_config()
cfg/FNN.json, RNN.json    Hyperparameters for the two backbones
cfg/RM42 Magnetic Brake/  Brake-specific data, firmware, test vectors, CAD
utils/
  data_processing.py      ADC → engineering units, filtering, segmentation
  model_generator.py      BlackboxModel, scheduled-sampling training loop
  model_executor.py       Closed-loop inference (NARX rollout)
  plotting.py             Comparison plots + test-vector plots
  test_vector.py          Parses .def files into exogenous-input DataFrames
outputs/<model>/          Created at runtime: Models/, Plots/, Training Data/,
                          Held Out/, Test Cases/
```

## Things to know before editing

- **`compute_feedback_features` is shared between train and inference.** Both
  `_rollout_sequence` (model_generator.py) and `predict` (model_executor.py)
  call it. If you change the feedback recipe, both paths get the change for
  free — but any divergence is a silent train/serve skew bug.
- **Hidden state and feedback buffer are per-rollout.** In `ModelExecutor.predict`
  they are local variables, not `self.*`. An earlier bug stored them on `self`
  and leaked state between calls; the comment at model_executor.py:62-64 calls
  this out. Keep them local.
- **PyTorch `nn.RNN` with `hidden=None`** is treated as a zero hidden state.
  That is intentional and matches the zero-seeded `y_history` buffer. Do not
  add a learned initial hidden state without also updating the inference path.
- **Splits are by experiment file, not by row.** `_split_sequences` shuffles
  CSV indices with a seeded RNG and assigns whole files to train / val / test.
  Val+test files are copied into `held_out_path` so they can be inspected
  later. Do not introduce row-level shuffling — it would leak across the
  autoregressive boundary.
- **Closed-loop validation only.** `_evaluate` runs with `teacher_forcing_prob=0`.
  The number printed each epoch is the metric the model will be judged on in
  deployment.
- **EMA-smoothed val loss drives all decisions.** LR scheduler, early stopping,
  and best-checkpoint selection all consume `decision_loss` (smoothed) rather
  than the raw val loss. The raw signal is intrinsically noisy on small
  datasets. Set `val_smoothing: 0.0` to disable.
- **Feedback noise (`feedback_noise_std`) is the single most useful knob against
  autoregressive drift.** Targets are normalised to ~[0, 1], so 0.01 is ~1%
  drift on the teacher-forced feedback. Don't remove it casually.
- **Gradient clipping defaults differ by backbone.** FNN uses 5.0, RNN uses 1.0.
  RNNs explode through time more easily.

## Running the app

```bash
python main.py
```

If only one config exists under `cfg/`, it's auto-selected. Otherwise the menu
prompts for one. The submenu options 1–8 are: preprocess raw data, train from
scratch, retrain existing checkpoint, compare predictions vs. experimental data,
execute a test vector, run all test cases (and produce a GIF), delete generated
files, exit.

The held-out test loss is printed automatically at the end of `train_model` —
no separate "run on test set" step is required.

## Formatting

Python is formatted with **black** (default config, line length 88). If you
edit a `.py` file, run `python3 -m black <file>` before finishing.

## Things that look weird but are intentional

- `config.py` uses module-level globals populated by `load_config()`. It is
  imported as `cfg` everywhere. Not the prettiest pattern, but it's the
  established one — don't refactor it into a class without a reason.
- `main.py:40` ASCII-art logo triggers a `SyntaxWarning` about `\ `. Pre-existing
  and harmless; leave it unless you want to convert the string to a raw literal.
- The `lr_scheduler.patience` field (4) is the **scheduler** patience, distinct
  from `early_stop_patience` (25). Both look at the EMA-smoothed val loss.
- `feedback_noise_std` defaults to 0.0 if missing from the JSON. The bundled
  configs may or may not set it explicitly — check `cfg/*.json` before
  attributing behaviour to it.
