# Config files

Each `*.json` here is a full training-run spec. They're not partial overrides — pick one at launch and it drives everything.

## Adding a new field

1. Add it to the relevant JSON(s).
2. Declare it as a module-level `None` in `config.py` at the top of the file.
3. Add a matching `global` line inside `load_config`.
4. **Read it via `config_data.get('field', default)`, not `config_data['field']`.** Older configs in the wild (and `cfg/RNN.json` vs `cfg/FNN.json`) must keep loading even if they don't have the new key.
5. Reference it as `cfg.field` from `utils/model_generator.py` etc.

The `load_config` function is the only place that does dict lookups against the JSON; everywhere else in the codebase imports `config as cfg` and reads attributes. Don't bypass this.

## Keep FNN and RNN in sync

When you change one config's schema, change the other. The two backbones share the training loop, so a field that exists in one but not the other will silently fall back to the default in `load_config` — easy to miss.

Hyperparameter *values* differ between FNN and RNN on purpose:

- RNN uses a smaller backbone (fewer layers, fewer neurons) because deep ReLU RNNs are unstable on small data.
- RNN uses `grad_clip: 1.0` (vs 5.0 for FNN) — RNNs are far more prone to exploding gradients through time.

Everything else should track between the two unless there's a specific reason.
