import math

import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Linear):
    def __init__(
            self,
            in_features,
            out_features,
            bias=True,
            lora_alpha=1.0,
            lora_dim=8,
            qkv=True,
            fan_in_fan_out=False,
            # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
            merge_weights=True,
            **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)

        self.lora_alpha = lora_alpha
        self.lora_dim = lora_dim
        self.qkv = qkv
        self.fan_in_fan_out = fan_in_fan_out
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights

        if qkv:
            assert out_features // in_features == 3
            self.lora_A_q = nn.Parameter(self.weight.new_zeros((lora_dim, in_features)))
            self.lora_B_q = nn.Parameter(self.weight.new_zeros((in_features, lora_dim)))
            self.lora_A_v = nn.Parameter(self.weight.new_zeros((lora_dim, in_features)))
            self.lora_B_v = nn.Parameter(self.weight.new_zeros((in_features, lora_dim)))
        else:
            self.lora_A = nn.Parameter(self.weight.new_zeros((lora_dim, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, lora_dim)))
        self.scaling = lora_alpha / lora_dim

        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)

        # initialize A the same way as the default for nn.Linear and B to zero
        if hasattr(self, 'lora_A_q'):
            nn.init.kaiming_uniform_(self.lora_A_q, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B_q)
            nn.init.kaiming_uniform_(self.lora_A_v, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B_v)
        if hasattr(self, 'lora_A'):
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode=True):
        def T(w):
            return w.T if self.fan_in_fan_out else w

        nn.Linear.train(self, mode)
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.qkv:
                self.weight.data[:self.in_features, :] -= T(self.lora_B_q @ self.lora_A_q) * self.scaling
                self.weight.data[-self.in_features:, :] -= T(self.lora_B_v @ self.lora_A_v) * self.scaling
            else:
                self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
            self.merged = False

    def eval(self):
        def T(w):
            return w.T if self.fan_in_fan_out else w

        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            if self.qkv:
                self.weight.data[:self.in_features, :] += T(self.lora_B_q @ self.lora_A_q) * self.scaling
                self.weight.data[-self.in_features:, :] += T(self.lora_B_v @ self.lora_A_v) * self.scaling
            else:
                self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
            self.merged = True

    def forward(self, x):
        def T(w):
            return w.T if self.fan_in_fan_out else w

        result = F.linear(x, T(self.weight), bias=self.bias)
        if not self.merged:
            if self.qkv:
                lora_q = (x @ self.lora_A_q.T @ self.lora_B_q.T) * self.scaling
                lora_v = (x @ self.lora_A_v.T @ self.lora_B_v.T) * self.scaling
                result[:, :, :self.in_features] += lora_q
                result[:, :, -self.in_features:] += lora_v
            else:
                lora = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
                result += lora

        return result

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, r={}, alpha={}, merged={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.lora_dim, self.lora_alpha, self.merged
        )

