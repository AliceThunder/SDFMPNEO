"""Current-format serialization for the fixed analytic response model."""
from __future__ import annotations


class _NetworkMetadataMixin:
    def to_metadata(self):
        return {
            "kind": self.kind,
            "format_version": self.format_version,
            "lambdas": self.lambdas.tolist(),
            "operating_names": list(self.operating_names),
            "max_response_time": self.max_response_time,
            "input_center": self.input_center.tolist(),
            "input_scale": self.input_scale.tolist(),
            "depth": self.depth,
            "channels_per_mode": self.channels_per_mode,
            "linear_rank": self.linear_rank,
            "hidden_rank": self.hidden_rank,
            "quadratic_rank": self.quadratic_rank,
            "square_rank": self.square_rank,
            "cross_rank": self.cross_rank,
            "state_rank": self.state_rank,
            "layer_widths": list(self.layer_widths),
            "layer_targets": [values.tolist() for values in self.layer_targets],
            "layer_hidden_ranks": list(self.layer_hidden_ranks),
            "layer_cross_ranks": list(self.layer_cross_ranks),
            "layer_state_ranks": list(self.layer_state_ranks),
            "state_feature_term_budget": self.state_feature_term_budget,
            "structure": self.structure_summary(0.0),
        }

    @classmethod
    def from_metadata(cls, metadata, parameters):
        if metadata.get("kind") != cls.kind:
            raise ValueError("unsupported fixed analytic response network kind")
        version = int(metadata.get("format_version", -1))
        if version not in {6, cls.format_version}:
            raise ValueError("unsupported fixed analytic response network format")
        kwargs = dict(
            max_response_time=metadata["max_response_time"],
            input_center=metadata["input_center"],
            input_scale=metadata["input_scale"],
            depth=metadata["depth"],
            channels_per_mode=metadata["channels_per_mode"],
            linear_rank=metadata["linear_rank"],
            hidden_rank=metadata["hidden_rank"],
            quadratic_rank=metadata["quadratic_rank"],
            square_rank=metadata["square_rank"],
            cross_rank=metadata["cross_rank"],
            state_rank=metadata["state_rank"],
            parameters=parameters,
        )
        # v6 checkpoints used one common width/rank per layer. Constructing the
        # network without v7 funnel fields reproduces that exact legacy layout,
        # so their parameter vectors remain loadable without migration.
        if version >= 7:
            kwargs.update(
                layer_widths=metadata.get("layer_widths"),
                layer_targets=metadata.get("layer_targets"),
                layer_hidden_ranks=metadata.get("layer_hidden_ranks"),
                layer_cross_ranks=metadata.get("layer_cross_ranks"),
                layer_state_ranks=metadata.get("layer_state_ranks"),
                state_feature_term_budget=metadata.get("state_feature_term_budget", 24),
            )
        return cls(metadata["lambdas"], metadata["operating_names"], **kwargs)


__all__ = ["_NetworkMetadataMixin"]
