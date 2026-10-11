from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256


_UINT64_RANGE = 1 << 64
GENERATION2_SPLIT_CONTRACT = 1


def _score(index: int, seed: int) -> int:
    payload = f"sdfmpneo-vnext-gen2-split:{int(seed)}:{int(index)}".encode("ascii")
    return int.from_bytes(sha256(payload).digest()[:8], "big")


@dataclass(frozen=True)
class Generation2Partition:
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    seed: int
    validation_fraction: float
    test_fraction: float

    def __post_init__(self):
        groups = (
            tuple(int(value) for value in self.train_indices),
            tuple(int(value) for value in self.validation_indices),
            tuple(int(value) for value in self.test_indices),
        )
        all_indices = groups[0] + groups[1] + groups[2]
        if len(set(all_indices)) != len(all_indices):
            raise ValueError("generation-2 partition contains duplicate indices")
        object.__setattr__(self, "train_indices", groups[0])
        object.__setattr__(self, "validation_indices", groups[1])
        object.__setattr__(self, "test_indices", groups[2])
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "validation_fraction", float(self.validation_fraction))
        object.__setattr__(self, "test_fraction", float(self.test_fraction))

    @property
    def count(self) -> int:
        return (
            len(self.train_indices)
            + len(self.validation_indices)
            + len(self.test_indices)
        )

    def subset(self, values, split: str):
        values = tuple(values)
        name = str(split).strip().lower()
        if name == "train":
            indices = self.train_indices
        elif name in {"validation", "val"}:
            indices = self.validation_indices
        elif name == "test":
            indices = self.test_indices
        else:
            raise ValueError("split must be train, validation, or test")
        if values and max(indices, default=-1) >= len(values):
            raise ValueError("partition indices exceed the supplied sequence")
        return tuple(values[index] for index in indices)

    def membership(self) -> dict[int, str]:
        result = {index: "train" for index in self.train_indices}
        result.update({index: "validation" for index in self.validation_indices})
        result.update({index: "test" for index in self.test_indices})
        return result

    def fingerprint(self) -> str:
        payload = (
            f"contract={GENERATION2_SPLIT_CONTRACT};seed={self.seed};"
            f"validation={self.validation_fraction:.17g};"
            f"test={self.test_fraction:.17g};count={self.count}"
        )
        return sha256(payload.encode("ascii")).hexdigest()


def generation2_partition(
    count: int,
    *,
    validation_fraction: float = 0.10,
    test_fraction: float = 0.10,
    seed: int = 2027,
) -> Generation2Partition:
    count = int(count)
    validation_fraction = float(validation_fraction)
    test_fraction = float(test_fraction)
    if count < 3:
        raise ValueError("generation-2 partition requires at least three samples")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie strictly between zero and one")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must lie strictly between zero and one")
    if validation_fraction + test_fraction >= 1.0:
        raise ValueError("validation_fraction + test_fraction must be less than one")

    validation_limit = int(validation_fraction * _UINT64_RANGE)
    test_limit = int((validation_fraction + test_fraction) * _UINT64_RANGE)

    train = []
    validation = []
    test = []
    for index in range(count):
        score = _score(index, seed)
        if score < validation_limit:
            validation.append(index)
        elif score < test_limit:
            test.append(index)
        else:
            train.append(index)

    if not train or not validation or not test:
        raise ValueError(
            "generation-2 hash partition produced an empty subset; increase the "
            "sample count or choose a different split seed"
        )

    return Generation2Partition(
        tuple(train),
        tuple(validation),
        tuple(test),
        int(seed),
        validation_fraction,
        test_fraction,
    )


__all__ = [
    "GENERATION2_SPLIT_CONTRACT",
    "Generation2Partition",
    "generation2_partition",
]
