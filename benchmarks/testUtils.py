import torch
import timeit
import unittest

from core.utils.Utils import clamp_update, soft_wta


class TestFP16Benchmarks(unittest.TestCase):

    def test_benchmark_fp16_ops(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\nBenchmark running on: {device}")

        x = torch.randn(4096, device=device, dtype=torch.float16)
        max_norm = torch.ones(1, device=device, dtype=torch.float16) * 18

        for _ in range(10):
            clamp_update(x, max_norm)
            soft_wta(x)

        runs = 10000

        t_clamp = timeit.timeit(
            stmt="clamp_update(x, max_norm)",
            number=runs,
            globals={"clamp_update": clamp_update, "x": x, "max_norm": max_norm}
        )

        t_wta = timeit.timeit(
            stmt="soft_wta(x)",
            number=runs,
            globals={"soft_wta": soft_wta, "x": x}
        )

        clamp_us = (t_clamp / runs) * 1e6
        wta_us = (t_wta / runs) * 1e6

        print(f"clamp_update: {t_clamp:.4f} sec for {runs} runs ({clamp_us:.2f} µs per call)")
        print(f"soft_wta:     {t_wta:.4f} sec for {runs} runs ({wta_us:.2f} µs per call)")

        self.assertTrue(t_clamp > 0)
        self.assertTrue(t_wta > 0)


if __name__ == "__main__":
    unittest.main()
