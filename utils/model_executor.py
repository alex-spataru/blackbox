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
import torch
import numpy as np
import config as cfg

from utils.model_generator import BlackboxModel, compute_feedback_features


class ModelExecutor:
    """
    @brief Loads a trained BlackboxModel and runs closed-loop predictions.

    Inference replicates exactly the feedback feature recipe used during
    training (via compute_feedback_features), so first- and higher-order
    derivatives stay consistent across train/inference.
    """

    def __init__(self):
        """
        Loads the best checkpoint from disk in eval mode.
        """
        model_path = os.path.join(cfg.model_save_path, f'{cfg.model_name}.pt')
        self.model = BlackboxModel().to(cfg.device)
        self.model.load_state_dict(
            torch.load(model_path, weights_only=True)
        )
        self.model.eval()

    def predict(self, df):
        """
        Closed-loop rollout on a DataFrame of exogenous inputs.

        @param df Pandas DataFrame; must contain every column in cfg.inputs.
        @return Dict mapping output name -> NumPy array of predictions.
        """
        output_dim = len(cfg.outputs)
        num_derivs = cfg.num_derivatives
        buffer_len = num_derivs + 1

        # Per-call state: fresh hidden state, fresh feedback buffer.
        # The original implementation kept self.hidden_state across calls,
        # which leaked one test's terminal state into the next test.
        hidden_state = None
        y_history = [
            torch.zeros(output_dim, dtype=torch.float32, device=cfg.device)
            for _ in range(buffer_len)
        ]

        # Pre-materialise the exogenous inputs once (avoids per-row .to_numpy()).
        exogenous = torch.tensor(
            df[cfg.inputs].values, dtype=torch.float32, device=cfg.device
        )

        T = exogenous.shape[0]
        preds_tensor = torch.zeros(
            T, output_dim, dtype=torch.float32, device=cfg.device
        )

        with torch.no_grad():
            for t in range(T):
                feedback_features = compute_feedback_features(
                    y_history, num_derivs
                )
                x_t = torch.cat([exogenous[t]] + feedback_features, dim=0)

                if cfg.rnn:
                    x_in = x_t.unsqueeze(0).unsqueeze(0)   # (1, 1, input_dim)
                    y_step, hidden_state = self.model(x_in, hidden_state)
                    y_step = y_step.squeeze(0).squeeze(0)
                else:
                    x_in = x_t.unsqueeze(0)                # (1, input_dim)
                    y_step = self.model(x_in).squeeze(0)

                preds_tensor[t] = y_step
                y_history = [y_step] + y_history[:-1]

        preds_np = preds_tensor.cpu().numpy()
        return {name: preds_np[:, i] for i, name in enumerate(cfg.outputs)}
