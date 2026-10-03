"""Resume-from-checkpoint test: an interrupted + resumed run must match an uninterrupted run exactly (CPU)."""
import pytest

from src.train import train
from src.utils import load_config

TEXT = ("the quick brown fox jumps over the lazy dog. " * 400)


def _cfg(tmp_path, name):
    cfg = load_config()
    cfg["model"].update(d_model=32, n_heads=2, n_layers=2, ffn_dim=64, context_length=16, dropout=0.1)
    cfg["train"].update(max_steps=30, eval_interval=10, save_interval=10, eval_batches=2, batch_size=4, log_interval=5,
                        sample_interval=30, sample_tokens=10, device="cpu",
                        out_dir=str(tmp_path / name / "out"), checkpoint_dir=str(tmp_path / name / "ckpt"))
    return cfg


def test_interrupted_run_resumes_and_matches_uninterrupted(tmp_path):
    ref = train(_cfg(tmp_path, "ref"), run_name="r", quiet=True, text=TEXT)

    cfg = _cfg(tmp_path, "int")
    with pytest.raises(InterruptedError):
        train(cfg, run_name="r", quiet=True, text=TEXT, resume=True, stop_after_step=17)   # "timeout" after step 17
    resume_file = tmp_path / "int" / "ckpt" / "r_resume.pt"
    assert resume_file.exists()                                                           # saved at step 10

    out = train(cfg, run_name="r", quiet=True, text=TEXT, resume=True)                    # reconnect and run again
    assert not resume_file.exists()                                                       # cleaned up after finishing
    assert out["final_val_loss"] == pytest.approx(ref["final_val_loss"], abs=1e-6)
    assert out["final_train_loss"] == pytest.approx(ref["final_train_loss"], abs=1e-6)


def test_resume_file_from_different_config_is_ignored(tmp_path):
    cfg = _cfg(tmp_path, "mix")
    with pytest.raises(InterruptedError):
        train(cfg, run_name="r", quiet=True, text=TEXT, resume=True, stop_after_step=12)
    cfg["train"]["max_steps"] = 20                                                        # different setup -> must restart
    out = train(cfg, run_name="r", quiet=True, text=TEXT, resume=True)
    assert out["train_config"]["max_steps"] == 20
