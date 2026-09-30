import pytest

from sdfmpneo_vnext.cli import (
    _require_mixed_reference,
    build_parser,
)
from sdfmpneo_vnext.__main__ import main


def test_vnext_cli_registers_complete_workflow():
    parser = build_parser()
    commands = {
        action.dest: action
        for action in parser._actions
        if action.dest == "command"
    }
    assert "command" in commands
    subcommands = set(
        commands[
            "command"
        ].choices
    )
    assert {
        "self-check",
        "dataset-generate",
        "hybrid-generate",
        "dataset-migrate",
        "train-port",
        "hybrid-train-port",
        "train-spatial",
        "hybrid-train-spatial",
        "audit-port",
        "audit-spatial",
        "active-learn",
        "calibrate",
        "bundle-publish",
        "release",
    }.issubset(
        subcommands
    )


def test_legacy_self_check_flag_is_translated(monkeypatch):
    captured = {}

    def fake_cli(argv):
        captured[
            "argv"
        ] = list(
            argv
        )
        return 0

    monkeypatch.setattr(
        "sdfmpneo_vnext.__main__._cli_main",
        fake_cli,
    )
    assert main(
        [
            "--self-check"
        ]
    ) == 0
    assert captured[
        "argv"
    ] == [
        "self-check"
    ]



def test_release_parser_exposes_certified_gate_controls():
    parser = build_parser()
    args = parser.parse_args(
        [
            "release",
            "dataset",
            "port.pt",
            "spatial.pt",
            "bundle",
        ]
    )
    assert args.certified_backend == "matrix_free"
    assert args.certified_coarse_segments == 8
    assert args.certified_fine_segments == 12
    assert args.certified_convergence_limit == 0.02
    assert args.certified_fast_correction_limit == 0.20
    assert args.certified_truth_limit == 0.02
    assert args.artifact_family == "auto"



def test_predict_parser_accepts_bundle_request_and_output():
    parser = build_parser()
    args = parser.parse_args(
        [
            "predict",
            "bundle",
            "request.json",
            "--output",
            "result.json",
        ]
    )
    assert str(args.bundle).endswith("bundle")
    assert str(args.request).endswith("request.json")
    assert str(args.output).endswith("result.json")
    assert args.device == "cpu"
    assert not args.allow_development_bundle



class _BackendDataset:
    def __init__(self, mapping):
        self.mapping = dict(mapping)

    def reference_backends(self, split):
        return tuple(
            self.mapping.get(
                split,
                (),
            )
        )


def test_production_workflows_require_pure_mixed_reference_splits():
    good = _BackendDataset(
        {
            "train": ("mixed",),
            "validation": ("mixed",),
            "release": ("mixed",),
        }
    )
    _require_mixed_reference(
        good,
        ("train", "validation", "release"),
        context="test",
    )

    mixed = _BackendDataset(
        {
            "train": ("mixed", "mqs"),
        }
    )
    with pytest.raises(
        SystemExit,
        match="pure mixed-reference train split",
    ):
        _require_mixed_reference(
            mixed,
            ("train",),
            context="test",
        )


def test_hybrid_generate_parser_exposes_core_design_domain_controls():
    parser = build_parser()
    args = parser.parse_args(
        [
            "hybrid-generate",
            "dataset",
            "--count",
            "9",
            "--dc-probability",
            "0.2",
            "--dc-conductive-probability",
            "0.6",
            "--package-count-min",
            "2",
            "--package-count-max",
            "4",
            "--nested-package-probability",
            "1.0",
            "--graded-package-probability",
            "0.75",
            "--lossy-background-probability",
            "0.4",
            "--package-mu-max",
            "2.5",
        ]
    )
    assert args.count == 9
    assert args.dc_probability == 0.2
    assert args.dc_conductive_probability == 0.6
    assert args.package_count_min == 2
    assert args.package_count_max == 4
    assert args.nested_package_probability == 1.0
    assert args.graded_package_probability == 0.75
    assert args.lossy_background_probability == 0.4
    assert args.package_mu_min == 1.0
    assert args.package_mu_max == 2.5
    assert args.handler.__name__ == "command_hybrid_generate"


def test_hybrid_training_parsers_bind_artifacts_and_batch_controls():
    parser = build_parser()
    port = parser.parse_args(
        [
            "hybrid-train-port",
            "dataset",
            "port.pt",
            "--batch-size",
            "6",
            "--device",
            "cuda",
        ]
    )
    assert str(
        port.dataset
    ).endswith(
        "dataset"
    )
    assert str(
        port.output
    ).endswith(
        "port.pt"
    )
    assert port.batch_size == 6
    assert port.device == "cuda"
    assert (
        port.handler.__name__
        == "command_hybrid_train_port"
    )

    spatial = parser.parse_args(
        [
            "hybrid-train-spatial",
            "dataset",
            "port.pt",
            "spatial.pt",
            "--batch-size",
            "5",
            "--background-radial-order",
            "14",
        ]
    )
    assert str(
        spatial.port_artifact
    ).endswith(
        "port.pt"
    )
    assert str(
        spatial.output
    ).endswith(
        "spatial.pt"
    )
    assert spatial.batch_size == 5
    assert spatial.background_radial_order == 14
    assert (
        spatial.handler.__name__
        == "command_hybrid_train_spatial"
    )


def test_bundle_publish_and_release_accept_explicit_hybrid_family():
    parser = build_parser()
    publish = parser.parse_args(
        [
            "bundle-publish",
            "port.pt",
            "bundle",
            "--artifact-family",
            "hybrid",
        ]
    )
    assert publish.artifact_family == "hybrid"

    release = parser.parse_args(
        [
            "release",
            "dataset",
            "port.pt",
            "spatial.pt",
            "bundle",
            "--artifact-family",
            "hybrid",
        ]
    )
    assert release.artifact_family == "hybrid"
    assert (
        release.handler.__name__
        == "command_release"
    )
