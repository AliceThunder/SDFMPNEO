from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from sdfmpneo_vnext import (
    ImmutableTeacherDataset,
    MQSConfig,
    sample_two_coil_mvp_scene,
)


def main():
    parser = argparse.ArgumentParser(
        description="Generate immutable vNext dense-teacher samples."
    )
    parser.add_argument("output", type=Path)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--baseline-segments", type=int, default=64)
    args = parser.parse_args()

    if args.count < 1:
        raise SystemExit("--count must be >= 1")

    if (args.output / "manifest.json").exists():
        dataset = ImmutableTeacherDataset(
            args.output
        )
    else:
        dataset = ImmutableTeacherDataset.create(
            args.output,
            split_seed=args.seed,
        )

    rng = np.random.default_rng(args.seed)
    teacher = MQSConfig(
        segments_per_turn=12,
        min_segments=16,
        section_degree=1,
        radial_order=4,
        angular_order=24,
        line_order=2,
    )
    for index in range(args.count):
        scene, frequency = (
            sample_two_coil_mvp_scene(rng)
        )
        record = dataset.generate_and_add(
            scene,
            frequency,
            teacher_config=teacher,
            baseline_segments=args.baseline_segments,
            reference_backend="mixed",
            source="initial",
        )
        print(
            f"[{index + 1:04d}/{args.count:04d}] "
            f"{record.sample_id[:12]} "
            f"{record.split} "
            f"{frequency / 1e3:.2f} kHz"
        )
    print("counts:", dataset.counts())


if __name__ == "__main__":
    main()
