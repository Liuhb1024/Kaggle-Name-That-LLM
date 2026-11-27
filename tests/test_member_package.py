import numpy as np
import pytest

from ntl_baseline.constants import LABELS
from ntl_member import package


def test_parity_uses_absolute_macro_recall_difference_and_log_probs():
    y = np.tile(np.arange(18), 4)
    logp = np.log(0.82 * np.eye(18)[y] + 0.01)
    targets = [LABELS[i] for i in y]
    result = package.check_parity(logp, logp, logp, targets)
    assert result["parity_mean_abs_logp"] == 0.0
    with pytest.raises(ValueError, match="parity"):
        package.check_parity(logp[:, ::-1], logp, logp, targets)
    with pytest.raises(ValueError, match="precision"):
        package.check_parity(logp, logp, logp[:, ::-1], targets)
    with pytest.raises(ValueError):
        package.check_parity(logp, np.full_like(logp, np.nan), logp, targets)


def test_adapter_keys_match_real_modules_and_have_both_factors():
    name = "layers.0.self_attn.q_proj"
    keys = [f"base_model.model.{name}.lora_{ab}.weight" for ab in "AB"]
    package.check_adapter_keys(keys, {name})
    for wrong in ([keys[0]], [key.replace("q_proj", "k_proj") for key in keys]):
        with pytest.raises(ValueError):
            package.check_adapter_keys(wrong, {name})


def test_manifest_requires_exact_files_and_hashes(tmp_path):
    from ntl_member.v31_bridge import verify_manifest

    (tmp_path / "NOTICE.md").write_text("license")
    with pytest.raises(ValueError):
        verify_manifest(tmp_path, "9" * 64)


def test_reference_rejects_silent_failure_and_reordered_output(tmp_path):
    import pandas as pd

    frame = pd.DataFrame({"id": ["s-a", "s-b"], "text": ["synthetic one", "synthetic two"]})
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    train = tmp_path / "official-fixture.csv"
    train.write_text("id,text,model,prompt_id\n")
    runner = scripts / "infer.py"
    runner.write_text("print('MEMBER FAILED q8pool')")
    member = tmp_path / "q8pool_f0"
    with pytest.raises(ValueError, match="adapters"):
        package.reference_predict(scripts, tmp_path, [member], train, frame, tmp_path / "failed")
    runner.write_text(
        "import sys, numpy as np\n"
        "print('l24 q8pool: adapter q8pool_f0 done')\n"
        "np.savez(sys.argv[sys.argv.index('--save_final_logits')+1],"
        f"ids=['s-b','s-a'],labels={list(LABELS)!r},logits=np.zeros((2,18)))\n"
    )
    with pytest.raises(ValueError, match="ids"):
        package.reference_predict(scripts, tmp_path, [member], train, frame, tmp_path / "reordered")
