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
