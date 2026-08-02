import unittest
import torch

from core.utils.Utils import clamp_update, soft_wta


class TestFP16Ops(unittest.TestCase):

    def test_clamp_update_basic(self):
        dw = torch.tensor([1, 2, 4], dtype=torch.int8)
        max_norm = torch.tensor([8], dtype=torch.int8)

        out = clamp_update(dw, max_norm)

        self.assertEqual(out.dtype, torch.float16)
        self.assertEqual(out.shape, dw.shape)

        dw16 = dw.to(torch.float16)
        max16 = max_norm.to(torch.float16)
        expected = dw16 * (max16 / (torch.abs(dw16) + 1.0))

        self.assertTrue(torch.allclose(out, expected, atol=1e-3))

    def test_clamp_update_zero(self):
        dw = torch.tensor([0, 0, 0], dtype=torch.int8)
        max_norm = torch.tensor([10], dtype=torch.int8)

        out = clamp_update(dw, max_norm)

        expected = torch.zeros_like(out)
        self.assertTrue(torch.allclose(out, expected, atol=1e-3))

    def test_soft_wta_basic(self):
        x = torch.tensor([1, -2, 3], dtype=torch.int8)

        out = soft_wta(x)

        self.assertEqual(out.dtype, torch.float16)
        self.assertEqual(out.shape, x.shape)

        x16 = x.to(torch.float16)
        relu = torch.relu(x16)
        s = relu.sum() + 1.0
        expected = relu / s

        self.assertTrue(torch.allclose(out, expected, atol=1e-3))

    def test_soft_wta_all_negative(self):
        x = torch.tensor([-5, -3, -1], dtype=torch.int8)

        out = soft_wta(x)

        expected = torch.zeros_like(out)
        self.assertTrue(torch.allclose(out, expected, atol=1e-3))

    def test_soft_wta_large_values(self):
        x = torch.tensor([100, 50, 25], dtype=torch.int8)

        out = soft_wta(x)

        x16 = x.to(torch.float16)
        relu = torch.relu(x16)
        s = relu.sum() + 1.0
        expected = relu / s

        self.assertTrue(torch.allclose(out, expected, atol=1e-3))


if __name__ == "__main__":
    unittest.main()
