# Trellis reference vectors

Fixed decode vectors for the trellis weight types, read by `test-tq2t`, `test-tqk` and `test-tqp`.
Each file holds 64 blocks of 128 weights. Every check against them is bit-exact, so they pin the format: a decoder
that changes any output value fails.

| file | type | block bytes | used by |
|---|---|---|---|
| `tq2t_ref.bin` | TQ2_T | 34 | `test-tq2t` |
| `tqk6_ref.bin` / `tqk7_ref.bin` | TQK6 / TQK7 | 26 / 30 | `test-tqk` |
| `tqp5_ref.bin` / `tqp6_ref.bin` / `tqp7_ref.bin` | TQP5 / TQP6 / TQP7 | 42 / 50 / 58 | `test-tqp` |

Layout, all little-endian: `u32 n_blocks`, then `n_blocks` packed blocks of the type, then `n_blocks x 128` f32 expected
decoded values.

Origin: the TQ2_T and TQK files were written by the reference implementation of the offline trellis encoder, the TQP files
by an independent implementation of the format (see the header of each test). Neither generator is in this tree, so the
files are not regenerated here. They change only when the format changes, and then together with the decoders and
`TQ_STATE_BITS`.
