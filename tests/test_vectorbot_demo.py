import subprocess
import sys


def test_vectorbot_scripted_demo_fake_mode_smoke(tmp_path) -> None:
    output = tmp_path / "routes.jsonl"
    transcript = tmp_path / "transcript.md"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_vectorbot_demo.py",
            "--scripted",
            "--fake",
            "--output",
            str(output),
            "--transcript-output",
            str(transcript),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    lines = output.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 10
    assert "MOVE_UP" in lines[0]
    assert transcript.exists()


def test_vectorbot_route_log_schema_fake_mode(tmp_path) -> None:
    output = tmp_path / "routes.jsonl"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_vectorbot_demo.py",
            "--scripted",
            "--fake",
            "--output",
            str(output),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    import json

    row = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
    assert {
        "input_text",
        "predicted_label",
        "accepted",
        "confidence",
        "top_probabilities",
        "vector_norm",
        "state_before",
        "state_after",
        "diff",
        "model_id",
        "feature_space",
    }.issubset(row)
