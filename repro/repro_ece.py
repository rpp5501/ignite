"""Reproduction script for pytorch/ignite issue #1009 — "Add Expected Calibration Error metrics".

This script documents two things:

* **Part A** — whether an ``ExpectedCalibrationError`` metric exists in ``ignite.metrics`` at all.
* **Part B** — the core defect of the only prior attempt (PR #3132): it accumulates every prediction
  with ``torch.cat`` and therefore retains ``O(N)`` memory, which leads to OOM on large datasets. The
  implementation added for this issue instead keeps a fixed number of per-bin aggregates, so its
  retained state is ``O(num_bins)`` regardless of ``N``.

Run from the repository root::

    PYTHONPATH=. python repro/repro_ece.py

This file is a development/reproduction aid for the contribution write-up; it is not part of the
library and is not intended to be merged upstream.
"""

import torch


def part_a() -> None:
    print("=== Part A: ExpectedCalibrationError present in ignite.metrics? ===")
    try:
        import ignite.metrics as metrics

        present = hasattr(metrics, "ExpectedCalibrationError")
    except Exception:  # pragma: no cover - defensive
        present = False
    print(present)
    print()


class _PR3132LikeAccumulator:
    """Mimics PR #3132: stores *every* confidence and correctness flag via ``torch.cat``."""

    def __init__(self) -> None:
        self.confidences = torch.empty(0, dtype=torch.float32)
        self.corrects = torch.empty(0, dtype=torch.bool)

    def update(self, conf: torch.Tensor, correct: torch.Tensor) -> None:
        self.confidences = torch.cat((self.confidences, conf))
        self.corrects = torch.cat((self.corrects, correct))

    def state_bytes(self) -> int:
        return (
            self.confidences.element_size() * self.confidences.nelement()
            + self.corrects.element_size() * self.corrects.nelement()
        )


def _proposed_state_bytes(metric) -> int:
    return sum(t.element_size() * t.nelement() for t in (metric._bin_correct, metric._bin_conf, metric._bin_count))


def part_b() -> None:
    from ignite.metrics import ExpectedCalibrationError

    batch_size = 4096
    n_classes = 10
    print("=== Part B: retained state size as dataset size N grows (batch=4096) ===")
    print(f"{'batches':>8}{'N':>12}{'PR#3132 state':>17}{'proposed state':>18}")

    for n_batches in (10, 100, 1000):
        pr3132 = _PR3132LikeAccumulator()
        proposed = ExpectedCalibrationError(num_bins=n_classes)
        n = 0
        for _ in range(n_batches):
            probs = torch.softmax(torch.randn(batch_size, n_classes), dim=1)
            y = torch.randint(0, n_classes, (batch_size,))
            conf, pred = probs.max(dim=1)
            pr3132.update(conf, pred.eq(y))
            proposed.update((probs, y))
            n += batch_size
        pr_mb = pr3132.state_bytes() / 1e6
        prop_mb = _proposed_state_bytes(proposed) / 1e6
        print(f"{n_batches:>8}{n:>12}{pr_mb:>14.2f} MB{prop_mb:>15.6f} MB")


if __name__ == "__main__":
    part_a()
    part_b()
