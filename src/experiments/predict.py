"""Thử artifact v2 bằng prompt mới; mặc định dùng bản chọn trên validation."""

import argparse
import json
from pathlib import Path

import joblib

from src.experiments.data import ROOT, sha256
from src.experiments.inference import route_prompts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--experiment", default=None, help="A/B_word/B/C/D_word/D; bỏ trống để dùng bản đã chọn")
    parser.add_argument("--prompt", required=True)
    args = parser.parse_args()
    if Path(args.run_name).name != args.run_name or args.run_name in (".", ".."):
        parser.error("run-name phải là một tên thư mục đơn")
    try:
        selection = json.loads((ROOT / "reports/experiments" / args.run_name / "selection.json").read_text())
        name = args.experiment or selection["selected_experiment"]
        if name not in selection["experiments"]:
            parser.error("Experiment không tồn tại trong run")
        info = selection["experiments"][name]
        path = ROOT / info["artifact"]
        if sha256(path) != info["artifact_sha256"]:
            raise ValueError("Artifact đã thay đổi sau selection")
        bundle = joblib.load(path)
        result = route_prompts(bundle, [args.prompt])
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps({
        "experiment": name, "target": bundle["experiment"]["target"],
        "estimates": {key: float(value[0]) for key, value in result["estimates"].items()},
        "routing_score": float(result["score"][0]), "threshold": bundle["selected_policy"]["threshold"],
        "tree_weight": bundle["selected_policy"]["tree_weight"], "model": str(result["model"][0]),
        "note": "quality: score là lợi ích kỳ vọng; routing: score là xác suất nhãn cũ. Chưa gọi LLM.",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
