# Native v3 actor binary A

All integers and float32 payload values are little-endian. The packed header is
`<8s23IQ32s32s32s32s` (236 bytes, no C/C++ padding):

| Offset | Type | Field |
|---:|---|---|
| 0 | char[8] | `KAGSV3A\0` |
| 8 | u32 | version = 1 |
| 12 | u32 | header bytes = 236 |
| 16 | u32 | endian tag = `0x01020304` |
| 20 | u32 | scalar type = 1 (float32) |
| 24 | u32 | normalization tensor count = 8 |
| 28 | u32 | parameter tensor count = 28 |
| 32..96 | 17 × u32 | classes, token capacity, token-continuous width, category count, context, observation, resource, model scale, projection, scalar, embedding, hidden, resource-hidden, event-embedding, cells, stages, previous-action count |
| 100 | u64 | payload bytes |
| 108 | byte[32] | source checkpoint SHA256 |
| 140 | byte[32] | training shard manifest SHA256 |
| 172 | byte[32] | contract SHA256 |
| 204 | byte[32] | payload SHA256 |

The payload has no descriptors or alignment gaps. It is raw C-contiguous
float32 in this exact order:

1. `context_mean`, `context_std`, `observation_mean`, `observation_std`,
   `observation_length_mean` (one scalar), `observation_length_std` (one
   scalar), `resource_mean`, `resource_std`.
2. `context.weight`, `context.bias`, `observation.weight`, `observation.bias`,
   `observation_length.weight`, `observation_length.bias`,
   `token_embeddings.0.weight` through `token_embeddings.6.weight`,
   `token.weight`, `token.bias`, `begin.weight`, `begin.bias`,
   `resource.weight`, `resource.bias`, `cell.weight`, `stage.weight`,
   `previous.weight`, `gru.weight_ih`, `gru.weight_hh`, `gru.bias_ih`,
   `gru.bias_hh`, `head.weight`, `head.bias`.

PyTorch GRU gate chunks retain their native `r,z,n` order. Shapes are derived
from the header dimensions and are checked by the exporter before any output is
written. The contract SHA256 is the SHA256 of the ASCII `CONTRACT_TEXT` in
`export_frozen_binary.py` (no trailing newline).
