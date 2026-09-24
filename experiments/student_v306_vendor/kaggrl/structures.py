"""The tokenizer's single runtime data type; training structs are omitted."""

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class EncodedObservation:
    continuous: torch.Tensor
    token_type: torch.Tensor
    category_a: torch.Tensor
    category_b: torch.Tensor
    category_c: torch.Tensor
    x: torch.Tensor
    y: torch.Tensor
    owner: torch.Tensor
    own_unit_token_indices: tuple[int, ...]

    @property
    def num_tokens(self) -> int:
        return int(self.continuous.shape[0])
