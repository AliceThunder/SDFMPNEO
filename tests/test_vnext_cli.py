from sdfmpneo_vnext.cli import build_parser
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
        "dataset-migrate",
        "train-port",
        "train-spatial",
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
