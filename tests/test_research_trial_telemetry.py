from sdfmpneo.training.monitor import TrainingMonitor, read_jsonl
from sdfmpneo.training.research_telemetry import trial_event


def test_trial_event_is_structured_and_ephemeral(tmp_path):
    log = tmp_path / "metrics.jsonl"
    with TrainingMonitor(log, interval=60.0) as monitor:
        trial_event(
            monitor,
            event="candidate",
            iteration=2,
            candidate=1,
            status="rejected",
            reason="max_filter_exceeded",
            predicted_max=0.03,
            predicted_weighted_rms=0.02,
            actual_rms=0.021,
            actual_max=0.041,
        )
    rows = read_jsonl(log)
    trial_rows = [row for row in rows if row.get("trial_event") == "candidate"]
    assert len(trial_rows) == 1
    row = trial_rows[0]
    assert row["trial_iteration"] == 2
    assert row["trial_candidate"] == 1
    assert row["trial_status"] == "rejected"
    assert row["trial_reason"] == "max_filter_exceeded"
    assert row["trial_actual_max"] == 0.041
    assert "trial_event" not in rows[-1]
