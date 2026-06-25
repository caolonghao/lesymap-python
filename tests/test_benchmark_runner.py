import json
import subprocess
import sys


def test_kernel_benchmark_runner_smoke(tmp_path):
    output = tmp_path / "kernels.json"
    completed = subprocess.run(
        [
            sys.executable,
            "benchmarks/run_benchmarks.py",
            "--suite",
            "kernels",
            "--repeat",
            "1",
            "--output",
            str(output),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "Wrote kernels benchmark results" in completed.stdout
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["result"]["suite"] == "kernels"
    assert payload["metadata"]["versions"]["numpy"]

    benchmark_names = {item["name"] for item in payload["result"]["benchmarks"]}
    assert {"bm", "ttest", "welch", "regression", "patch", "fdr"} <= benchmark_names
    assert all(item["wall_time_seconds"] for item in payload["result"]["benchmarks"])
