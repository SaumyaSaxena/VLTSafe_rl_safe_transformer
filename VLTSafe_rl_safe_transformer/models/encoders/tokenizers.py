from VLTSafe_rl_safe_transformer.models.RARL.utils import mlp, get_activation
import torch.nn as nn
from typing import Sequence

class LowdimObsTokenizer(nn.Module):
    def __init__(self,
        input_shape,
        token_embedding_size: int = 128,
        hidden_sizes: Sequence[int] = [128, 128],
        obs_stack_keys: Sequence[str] = tuple(),
        activation: str = 'silu',
        output_activation: str = 'silu',
        device: str = 'cuda',
    ):
        self.mlp = mlp(
            [input_shape[-1]] + list(hidden_sizes) + [token_embedding_size], 
            get_activation(activation), output_activation=get_activation(output_activation)
        ).to(device)

    def forward(self, observations):
        return self.mlp(observations)



