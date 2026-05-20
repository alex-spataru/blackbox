# Project context for Claude

## What this is

A NARX (Nonlinear AutoRegressive with eXogenous inputs) dynamical-system modelling pipeline. Two backbones — an FNN and a multi-layer ReLU RNN — share the same scheduled-sampling closed-loop training loop. The original target is the RM42 magnetic brake; the pipeline itself is data-agnostic and configurable via JSON.

See `README.md` for the user-facing explanation. This file is for *how to work in this repo*, not what it does.

## Workflow

- **Branch + PR for non-trivial work.** Don't push straight to `main` even though the historical log does. A solo direct commit is fine for one-line fixes; anything that touches the training loop, config schema, or data pipeline goes through a branch and a PR.
- The user runs training interactively via `python main.py`. **You do not need to run the training loop yourself to validate changes.** Use `python -c "import config; config.load_config('cfg/FNN.json'); import utils.model_generator"` for a fast sanity check.
- Development happens on Windows. Watch the 260-char path limit — `outputs/<model>/Plots/...` can stack up. If a script breaks with a path error, suspect that first.
- The shell available here is PowerShell. Use PowerShell syntax in Bash-tool commands unless you specifically need POSIX behaviour.

## Invariants you must not break

1. **Train and inference share one feedback recipe.** Both `_rollout_sequence` (training) and `ModelExecutor.predict` (inference) build features via `compute_feedback_features` in `utils/model_generator.py`. If you change the feature recipe in one place, change both, or train/eval will silently diverge.
2. **Splits are by experiment, not by row.** Each CSV in `training_data_path` is one experimental run and lives in exactly one of train / val / test. The split is driven by `cfg.seed` and recorded in `Models/split_manifest.json`. Never concatenate sequences before splitting.
3. **Closed-loop validation uses zero teacher forcing.** `_evaluate` runs with `teacher_forcing_prob=0.0`. Do not "fix" the val noise by injecting teacher forcing into eval — that would make the metric a lie.
4. **Predictions fed back during training are detached.** `next_feedback = y_step.detach()` in `_rollout_sequence`. BPTT through the autoregressive feedback path is intentionally cut so the autograd graph doesn't grow with sequence length T. If you remove the detach, memory usage explodes on long sequences.
5. **`feedback_noise_std` only applies to teacher-forced feedback during training.** Don't apply it during inference or to free-running predictions — it's there to make the model robust *to* drift, not to inject drift at deployment.
6. **Normalization is one-way and lives in `data_processing.py`.** Training data is divided by `normalization_parameters` once at preprocessing time. The model operates entirely in normalised space; predictions are returned in normalised space. Don't add denormalisation anywhere in the model or executor — it's the caller's responsibility if they want engineering units.

## Where to make changes

| Change | File |
|---|---|
| Hyperparameters | `cfg/FNN.json` or `cfg/RNN.json`. Both configs should evolve together; keep them in sync conceptually. |
| Add a new config field | `config.py:load_config`. Use `config_data.get('field', default)` so older configs keep loading. See `cfg/CLAUDE.md`. |
| Model definition | `BlackboxModel` in `utils/model_generator.py`. Both FNN and RNN branches live here. |
| Training loop | `train_model` and `_rollout_sequence` in `utils/model_generator.py`. |
| Inference | `ModelExecutor.predict` in `utils/model_executor.py`. Mirror any change to the training rollout. |
| Data prep | `utils/data_processing.py` — adapt this for new datasets. |
| Loss function | `derivative_aware_loss` in `utils/model_generator.py`. Operates on `(T, output_dim)` tensors along the time axis. |

## Diagnosing a bad training run

The val curve is *intrinsically* noisy on small data because closed-loop rollouts are sensitive to small weight changes. Read the **smoothed** val column, not the raw one. If even the smoothed line is jagged, the model is autoregressively unstable. In rough order of effectiveness:

1. Raise `feedback_noise_std` (e.g. 0.01 → 0.02). Single biggest knob against drift.
2. Lower `scheduled_sampling.end_prob` toward 0 and/or `decay_epochs` so the model practices closed-loop earlier.
3. Raise `val_smoothing` (e.g. 0.7 → 0.8) if the LR scheduler is reacting to noise.
4. Lower `grad_clip` (especially for the RNN backbone).
5. *Then* consider capacity or regularisation changes. Reaching for more neurons before stabilising the feedback path is the classic mistake.

If you can't reproduce a result, check `split_manifest.json` — `cfg.seed` controls the split, but if someone added new CSVs the split membership changes.

## Things never to commit

- `outputs/` — runtime artefacts (checkpoints, plots, training history). They regenerate.
- `.claude/` — local tooling state.
- Anything containing raw experimental data that isn't already under `cfg/RM42 Magnetic Brake/Experimental Data`.
