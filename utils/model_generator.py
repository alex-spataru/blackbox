#
# Copyright (c) 2023 Alex Spataru <https://github.com/alex-spataru>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the 'Software'), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

import os
import json
import random
import shutil

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import config as cfg

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

#-------------------------------------------------------------------------------
# Module-level constants and helpers
#-------------------------------------------------------------------------------

# Each output contributes (feedback + N derivatives) extra input features.
def _additional_features():
    return 1 + cfg.num_derivatives

def set_seed(seed):
    """
    Make a training run reproducible across PyTorch, NumPy and Python's RNG.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

#-------------------------------------------------------------------------------
# Feedback / derivative feature computation
#-------------------------------------------------------------------------------

def compute_feedback_features(y_history, num_derivatives):
    """
    Build the autoregressive features (feedback + cascaded finite differences)
    from the most recent output values. This is used identically during
    training and inference so the model sees the same feature recipe at both
    times.

    @param y_history    List/tuple of past output tensors, newest first.
                        Length must be >= num_derivatives + 1. Each entry has
                        shape (output_dim,).
    @param num_derivatives Number of finite-difference derivatives to compute.

    @return List of tensors [feedback, d1, d2, ...] each shape (output_dim,).
    """
    if len(y_history) < num_derivatives + 1:
        raise ValueError(
            f'y_history needs at least {num_derivatives + 1} entries, '
            f'got {len(y_history)}'
        )

    features = [y_history[0]]
    current_level = list(y_history)
    for _ in range(num_derivatives):
        current_level = [
            current_level[i] - current_level[i + 1]
            for i in range(len(current_level) - 1)
        ]
        features.append(current_level[0])
    return features

#-------------------------------------------------------------------------------
# Neural model for predicting the next state of a dynamical system.
#-------------------------------------------------------------------------------

class BlackboxModel(nn.Module):
    """
    @brief Predicts the next-step output of a dynamical system from exogenous
           inputs and a recursive feedback of its own previous predictions
           (plus their finite-difference derivatives).
    """

    def __init__(self):
        super(BlackboxModel, self).__init__()

        input_size = len(cfg.inputs) + _additional_features() * len(cfg.outputs)
        output_size = len(cfg.outputs)
        hidden_layers = cfg.hidden_layers
        neurons_per_layer = cfg.neurons_per_layer

        if cfg.rnn:
            self.rnn = nn.RNN(
                input_size,
                neurons_per_layer,
                hidden_layers,
                nonlinearity='relu',
                batch_first=True,
                dropout=cfg.dropout_rate if hidden_layers > 1 else 0.0,
            )
            self.fc = nn.Linear(neurons_per_layer, output_size)

            for name, param in self.rnn.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    nn.init.zeros_(param.data)
            nn.init.xavier_uniform_(self.fc.weight)
            nn.init.zeros_(self.fc.bias)

        else:
            layers = [nn.Linear(input_size, neurons_per_layer), nn.ReLU()]
            for _ in range(hidden_layers):
                layers.append(nn.Linear(neurons_per_layer, neurons_per_layer))
                layers.append(nn.ReLU())
                if cfg.dropout_rate > 0:
                    layers.append(nn.Dropout(cfg.dropout_rate))
            layers.append(nn.Linear(neurons_per_layer, output_size))
            self.model = nn.Sequential(*layers)

            for layer in self.model:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def forward(self, x, hidden=None):
        """
        @param x      For FNN: shape (B, input_dim). For RNN: (B, T, input_dim).
        @param hidden RNN hidden state, or None.
        @return       FNN: (B, output_dim). RNN: ((B, T, output_dim), hidden).
        """
        if cfg.rnn:
            out, hidden = self.rnn(x, hidden)
            return self.fc(out), hidden
        return self.model(x)

#-------------------------------------------------------------------------------
# Loss function — finite differences along the TIME axis of a sequence
#-------------------------------------------------------------------------------

def derivative_aware_loss(y_pred, y_true, num_derivatives):
    """
    MSE between predictions and ground truth, plus MSE on successive
    time-derivatives. Both tensors are shape (T, output_dim) where dim 0 is
    time. This penalises both pointwise error and shape mismatches.
    """
    loss = F.mse_loss(y_pred, y_true)
    dp, dt = y_pred, y_true
    for _ in range(num_derivatives):
        if dp.shape[0] < 2:
            break
        dp = dp[1:] - dp[:-1]
        dt = dt[1:] - dt[:-1]
        loss = loss + F.mse_loss(dp, dt)
    return loss

#-------------------------------------------------------------------------------
# Sequence loading and train/val/test split
#-------------------------------------------------------------------------------

class Sequence:
    """
    One experimental run, materialised as device-resident tensors.

    X_exo: (T, len(cfg.inputs))  -- exogenous inputs only (no feedback)
    y:     (T, len(cfg.outputs))
    """
    __slots__ = ('name', 'X_exo', 'y')

    def __init__(self, name, X_exo, y):
        self.name = name
        self.X_exo = X_exo
        self.y = y

    def __len__(self):
        return self.X_exo.shape[0]


def _load_sequences():
    """
    Load every CSV in cfg.training_data_path as an independent Sequence.
    Feedback features are NOT materialised here -- they are rebuilt on the fly
    during training so we can mix teacher-forced and free-running steps.
    """
    files = sorted(
        f for f in os.listdir(cfg.training_data_path) if f.endswith('.csv')
    )
    if not files:
        raise RuntimeError(
            f'No CSVs found in {cfg.training_data_path}; '
            f'run "Process experimental data" first.'
        )

    sequences = []
    for fname in files:
        path = os.path.join(cfg.training_data_path, fname)
        df = pd.read_csv(path)

        missing_in = [c for c in cfg.inputs if c not in df.columns]
        missing_out = [c for c in cfg.outputs if c not in df.columns]
        if missing_in or missing_out:
            print(
                f'-> Skipping {fname}: missing columns '
                f'in={missing_in} out={missing_out}'
            )
            continue

        X_exo = torch.tensor(
            df[cfg.inputs].values, dtype=torch.float32, device=cfg.device
        )
        y = torch.tensor(
            df[cfg.outputs].values, dtype=torch.float32, device=cfg.device
        )
        sequences.append(Sequence(fname, X_exo, y))

    return sequences


def _split_sequences(sequences):
    """
    Split by *file*, not by row, so each experimental run lives in exactly one
    of train / val / test. With small datasets we still guarantee at least one
    sequence per non-zero-ratio split when possible.
    """
    n = len(sequences)
    if n < 3:
        raise RuntimeError(
            f'Need at least 3 experiment CSVs for train/val/test split, '
            f'found {n}. Lower val_ratio/test_ratio or gather more data.'
        )

    # Deterministic shuffle order
    rng = random.Random(cfg.seed)
    order = list(range(n))
    rng.shuffle(order)

    n_val = max(1, int(round(n * cfg.val_ratio)))
    n_test = max(1, int(round(n * cfg.test_ratio)))
    if n_val + n_test >= n:
        # Pathological config; fall back to a minimal split
        n_val, n_test = 1, 1

    test_idx = set(order[:n_test])
    val_idx = set(order[n_test:n_test + n_val])

    train, val, test = [], [], []
    for i, seq in enumerate(sequences):
        if i in test_idx:
            test.append(seq)
        elif i in val_idx:
            val.append(seq)
        else:
            train.append(seq)

    return train, val, test


def _write_split_manifest(train, val, test):
    """
    Record which CSVs went to which split, and copy the held-out files so the
    user can run predictions on data the model never saw during training.
    """
    os.makedirs(cfg.model_save_path, exist_ok=True)
    manifest_path = os.path.join(cfg.model_save_path, 'split_manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(
            {
                'seed': cfg.seed,
                'train': [s.name for s in train],
                'val':   [s.name for s in val],
                'test':  [s.name for s in test],
            },
            f, indent=2,
        )

    os.makedirs(cfg.held_out_path, exist_ok=True)
    for seq in val + test:
        src = os.path.join(cfg.training_data_path, seq.name)
        dst = os.path.join(cfg.held_out_path, seq.name)
        if os.path.exists(src):
            shutil.copyfile(src, dst)

    print(f'-> Split manifest: {manifest_path}')
    print(f'-> Train: {len(train)}, Val: {len(val)}, Test: {len(test)}')

#-------------------------------------------------------------------------------
# Sequence rollout with scheduled sampling
#-------------------------------------------------------------------------------

def _teacher_forcing_prob(epoch):
    """
    Linearly decay the teacher-forcing probability from start_prob down to
    end_prob over decay_epochs, then hold at end_prob.
    """
    ss = cfg.scheduled_sampling
    start = float(ss['start_prob'])
    end = float(ss['end_prob'])
    decay = max(1, int(ss['decay_epochs']))
    if epoch >= decay:
        return end
    return start + (end - start) * (epoch / decay)


def _rollout_sequence(model, seq, teacher_forcing_prob, train_mode):
    """
    Step through a sequence one timestep at a time, mixing teacher-forced and
    free-running feedback per step according to teacher_forcing_prob.

    During training (train_mode=True) we build an autograd graph through every
    step so gradients flow into the model from both the pointwise loss and the
    autoregressive feedback path. Predictions used as feedback are detached so
    BPTT does not explode through long sequences.

    @return Tensor of shape (T, output_dim) with the model's predictions.
    """
    T = len(seq)
    out_dim = seq.y.shape[1]
    num_derivs = cfg.num_derivatives
    buffer_len = num_derivs + 1

    # y_history[0] is the most-recent feedback. We seed with zeros, which
    # matches the .fillna(0) padding used by the original implementation.
    y_history = [
        torch.zeros(out_dim, dtype=torch.float32, device=cfg.device)
        for _ in range(buffer_len)
    ]

    hidden = None
    preds = []

    for t in range(T):
        feedback_features = compute_feedback_features(y_history, num_derivs)
        # Concatenate exogenous inputs with the feedback features
        x_t = torch.cat([seq.X_exo[t]] + feedback_features, dim=0)

        if cfg.rnn:
            # (B=1, T=1, input_dim)
            x_in = x_t.unsqueeze(0).unsqueeze(0)
            y_step, hidden = model(x_in, hidden)
            y_step = y_step.squeeze(0).squeeze(0)
        else:
            # (B=1, input_dim)
            x_in = x_t.unsqueeze(0)
            y_step = model(x_in).squeeze(0)

        preds.append(y_step)

        # Decide what to feed back at t+1: ground truth (teacher) or
        # the model's own prediction (student). Detach the prediction so the
        # autograd graph does not grow unbounded over T steps.
        if t + 1 < T:
            use_teacher = (
                train_mode and random.random() < teacher_forcing_prob
            )
            if use_teacher:
                next_feedback = seq.y[t]
                # Perturb the teacher signal so the model learns to tolerate
                # imperfect feedback. Without this it overfits to a clean
                # autoregressive input and explodes once it has to consume
                # its own (slightly wrong) predictions at eval time.
                noise_std = cfg.feedback_noise_std
                if noise_std > 0.0:
                    next_feedback = next_feedback + torch.randn_like(
                        next_feedback
                    ) * noise_std
            else:
                next_feedback = y_step.detach() if train_mode else y_step

            y_history = [next_feedback] + y_history[:-1]

    return torch.stack(preds, dim=0)

#-------------------------------------------------------------------------------
# Train / evaluate
#-------------------------------------------------------------------------------

def _evaluate(model, sequences):
    """
    Closed-loop validation: zero teacher forcing, no gradients.
    Returns the mean per-step MSE across all evaluation sequences.
    """
    model.eval()
    total_sq_err = 0.0
    total_steps = 0
    with torch.no_grad():
        for seq in sequences:
            preds = _rollout_sequence(model, seq, 0.0, train_mode=False)
            total_sq_err += F.mse_loss(
                preds, seq.y, reduction='sum'
            ).item()
            total_steps += seq.y.numel()
    return total_sq_err / max(1, total_steps)


def _save_state(model, name):
    os.makedirs(cfg.model_save_path, exist_ok=True)
    torch.save(
        model.state_dict(),
        os.path.join(cfg.model_save_path, name),
    )


def train_model(model):
    """
    Train the supplied BlackboxModel with scheduled-sampling closed-loop
    rollouts. Best model is selected on validation loss; LR is decayed on
    validation-loss plateaus.
    """
    set_seed(cfg.seed)

    sequences = _load_sequences()
    train_seqs, val_seqs, test_seqs = _split_sequences(sequences)
    _write_split_manifest(train_seqs, val_seqs, test_seqs)

    if not train_seqs:
        raise RuntimeError('No training sequences after split.')

    optimizer = Adam(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=cfg.lr_scheduler['factor'],
        patience=cfg.lr_scheduler['patience'],
        min_lr=cfg.lr_scheduler['min_lr'],
    )

    batch_size = max(1, int(cfg.batch_size))
    grad_clip = float(cfg.grad_clip)
    # EMA factor for smoothing val loss: smoothed = alpha*smoothed + (1-alpha)*raw
    # Higher alpha = heavier smoothing. 0.0 falls back to raw val loss.
    val_alpha = max(0.0, min(0.99, float(cfg.val_smoothing)))
    smoothed_val = None
    best_val_loss = float('inf')
    best_epoch = 0
    epochs_since_improvement = 0
    history = {}

    header_smoothed = f' | {"Val (EMA)":>12}' if val_alpha > 0 else ''
    print(
        f'{"Epoch":>5} | {"TF prob":>7} | '
        f'{"Train Loss":>12} | {"Val Loss":>12}{header_smoothed} | {"LR":>10}'
    )
    print('-' * (64 + (len(header_smoothed))))

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        tf_prob = _teacher_forcing_prob(epoch - 1)

        # Shuffle training-sequence order each epoch
        order = list(range(len(train_seqs)))
        random.shuffle(order)

        running_loss = 0.0
        running_count = 0
        optimizer.zero_grad()
        accumulated = 0

        for step, seq_idx in enumerate(order, start=1):
            seq = train_seqs[seq_idx]
            preds = _rollout_sequence(model, seq, tf_prob, train_mode=True)
            loss = derivative_aware_loss(preds, seq.y, cfg.num_derivatives)

            # Gradient accumulation: average over the mini-batch of sequences
            (loss / batch_size).backward()
            accumulated += 1
            running_loss += loss.item()
            running_count += 1

            if accumulated >= batch_size or step == len(order):
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                optimizer.zero_grad()
                accumulated = 0

        train_loss = running_loss / max(1, running_count)
        val_loss = _evaluate(model, val_seqs)

        # EMA-smooth the val loss before using it to make decisions. The raw
        # closed-loop val signal is intrinsically noisy on small datasets, and
        # reacting to it directly causes premature LR cuts and unlucky
        # best-checkpoint selection.
        if val_alpha > 0:
            smoothed_val = (
                val_loss if smoothed_val is None
                else val_alpha * smoothed_val + (1 - val_alpha) * val_loss
            )
            decision_loss = smoothed_val
        else:
            decision_loss = val_loss

        scheduler.step(decision_loss)
        current_lr = optimizer.param_groups[0]['lr']

        history[epoch] = {
            'train': train_loss,
            'val': val_loss,
            'val_smoothed': smoothed_val,
        }
        if val_alpha > 0:
            print(
                f'{epoch:5d} | {tf_prob:7.3f} | '
                f'{train_loss:12.6e} | {val_loss:12.6e} | '
                f'{smoothed_val:12.6e} | {current_lr:10.2e}'
            )
        else:
            print(
                f'{epoch:5d} | {tf_prob:7.3f} | '
                f'{train_loss:12.6e} | {val_loss:12.6e} | {current_lr:10.2e}'
            )

        # Early-stopping & best-model bookkeeping. Use the smoothed signal so
        # we don't latch onto a noisy outlier epoch as the "best" model.
        if best_val_loss - decision_loss > cfg.early_stop_threshold:
            best_val_loss = decision_loss
            best_epoch = epoch
            epochs_since_improvement = 0
            _save_state(model, f'{cfg.model_name}.pt')
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= cfg.early_stop_patience:
                print(
                    f'\n-> Early stopping at epoch {epoch}; '
                    f'best val loss {best_val_loss:.6e} at epoch {best_epoch}.'
                )
                break

    print(f'\n-> Best epoch: {best_epoch}, val loss: {best_val_loss:.6e}')

    # Final held-out evaluation with the best checkpoint
    best_path = os.path.join(cfg.model_save_path, f'{cfg.model_name}.pt')
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, weights_only=True))

    if test_seqs:
        test_loss = _evaluate(model, test_seqs)
        print(f'-> Held-out test loss: {test_loss:.6e}')

    history_path = os.path.join(cfg.model_save_path, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(
            {
                'best_epoch': best_epoch,
                'best_val_loss': best_val_loss,
                'history': history,
            },
            f, indent=2,
        )

#-------------------------------------------------------------------------------
# Public API used by main.py
#-------------------------------------------------------------------------------

def generate_models():
    """
    @brief Train a fresh BlackboxModel from scratch.
    """
    set_seed(cfg.seed)
    model = BlackboxModel().to(cfg.device)
    train_model(model)


def retrain_models():
    """
    @brief Continue training the existing best checkpoint on disk.
    """
    set_seed(cfg.seed)
    model = BlackboxModel().to(cfg.device)
    model_path = os.path.join(cfg.model_save_path, f'{cfg.model_name}.pt')
    model.load_state_dict(torch.load(model_path, weights_only=True))
    train_model(model)
