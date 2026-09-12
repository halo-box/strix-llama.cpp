#!/usr/bin/env python3

import os
import sys
import unittest
from pathlib import Path

if "NO_LOCAL_GGUF" not in os.environ and (Path(__file__).parent.parent.parent / "gguf-py").exists():
    sys.path.insert(0, str(Path(__file__).parent.parent))

from gguf.constants import MODEL_ARCH, MODEL_ARCH_NAMES, MODEL_TENSOR, MODEL_TENSORS, TENSOR_NAMES


class TestDeepSeek41Schema(unittest.TestCase):

    def test_architecture_name(self):
        self.assertEqual(MODEL_ARCH_NAMES[MODEL_ARCH.DEEPSEEK41], "deepseek41")

    def test_engram_tensor_names(self):
        expected = {
            MODEL_TENSOR.ENGRAM_EMBD: "blk.14.engram_embd",
            MODEL_TENSOR.ENGRAM_Q_NORM: "blk.14.engram_q_norm",
            MODEL_TENSOR.ENGRAM_K_NORM: "blk.14.engram_k_norm",
            MODEL_TENSOR.ENGRAM_KV: "blk.14.engram_kv",
        }

        for tensor, name in expected.items():
            self.assertIn(tensor, MODEL_TENSORS[MODEL_ARCH.DEEPSEEK41])
            self.assertEqual(TENSOR_NAMES[tensor].format(bid=14), name)

    def test_output_head_does_not_require_deepseek4_hc_tensors(self):
        tensors = MODEL_TENSORS[MODEL_ARCH.DEEPSEEK41]

        self.assertIn(MODEL_TENSOR.OUTPUT_NORM, tensors)
        self.assertIn(MODEL_TENSOR.OUTPUT, tensors)
        self.assertNotIn(MODEL_TENSOR.HC_HEAD_FN, tensors)
        self.assertNotIn(MODEL_TENSOR.HC_HEAD_BASE, tensors)
        self.assertNotIn(MODEL_TENSOR.HC_HEAD_SCALE, tensors)


if __name__ == "__main__":
    unittest.main()
