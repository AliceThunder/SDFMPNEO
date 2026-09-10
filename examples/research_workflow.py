"""Complete fixed-network training -> save -> reload -> inference -> independent check."""
from pathlib import Path
import json

from sdfmpneo import ResearchElectroThermalModel, ResearchTrainingConfig, demo_research_model
from sdfmpneo.__main__ import jsonable


def main():
    output = Path("results/research_demo")
    output.mkdir(parents=True, exist_ok=True)
    model = demo_research_model()
    rank = model.core.thermal_model.rank
    config = ResearchTrainingConfig(
        initial_lower=(0.0,) * rank,
        initial_upper=(2.0,) * rank,
        operating_lower=(1000.0, 0.0),
        operating_upper=(3000.0, 1000.0),
        time_horizon=0.25,
        residual_tolerance=0.002,
        sample_count=16,
        validation_count=16,
        max_network_depth=3,
        max_channels_per_mode=2,
    )
    report = model.train(
        config,
        progress=lambda iteration, rms, maximum: print(
            f"iteration={iteration} rms={rms:.6g} maximum={maximum:.6g}", flush=True),
    )
    model.save(output / "model.npz")
    model = ResearchElectroThermalModel.load(output / "model.npz")
    prediction = model.predict(0.2, a0=[1.0] * rank, operating=[2000.0, 500.0])
    validation = model.validate_trajectory(
        [0.0, 0.05, 0.1, 0.2, 0.25], a0=[1.0] * rank, operating=[2000.0, 500.0])
    result = {"training": report, "prediction": prediction, "validation": validation}
    (output / "results.json").write_text(
        json.dumps(jsonable(result), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("temperature max (K):", prediction["maximum_temperature"])
    print("independent maximum temperature error (K):", validation["maximum_temperature_error"])
    if not report.numerical_tolerance_met:
        raise SystemExit("training did not reach the requested residual tolerance; inspect results.json")
    print(output / "model.npz")


if __name__ == "__main__":
    main()
