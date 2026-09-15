import sdfmpneo.unified_corrected_physics_gate as physics_gate
import sdfmpneo.unified_corrected_truth_preflight as preflight
import sdfmpneo.unified_runtime as runtime


MODEL = "canonical_local_full_fine_minus_coarse_self_defect_v3"


def test_localized_self_model_metadata_is_consistent_across_release_gates():
    settings = {
        "BACKGROUND": {
            "fine_step": 0.012,
            "mesh_check": {"refinement_factor": 0.75},
        }
    }
    skipped = preflight._skipped_mesh_report(settings, "test")

    assert preflight._SELF_CORRECTION_MODEL == MODEL
    assert physics_gate._SELF_CORRECTION_MODEL == MODEL
    assert runtime._SELF_CORRECTION_MODEL == MODEL
    assert skipped["self_correction_model"] == MODEL
