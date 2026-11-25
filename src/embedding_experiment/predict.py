"""CLI dùng router embedding đã lưu, không huấn luyện hoặc gọi LLM."""

import argparse
import json

import joblib

from src.embedding_experiment.common import MODES, ROOT, VARIANTS, run_dirs, sha256
from src.embedding_experiment.encoder import FrozenEncoder
from src.embedding_experiment.inference import predict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="minilm_2026-11-25")
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--mode", choices=list(MODES), default="tuned_router")
    parser.add_argument("--prompt", required=True)
    args = parser.parse_args()
    try:
        reports, _, _ = run_dirs(args.run_name)
        selection = json.loads((reports / "selection.json").read_text())
        artifact = ROOT / selection["artifact"]
        if sha256(artifact) != selection["artifact_sha256"]:
            raise ValueError("Artifact đã thay đổi")
        bundle = joblib.load(artifact)
        variant = args.variant or bundle["selected_variant"]
        encoder = FrozenEncoder() if variant != "structure" else None
        result = predict(bundle, [args.prompt], variant, args.mode, encoder)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps({"variant": variant, "mode": args.mode,
                      "tree_probability": float(result["tree_probability"][0]),
                      "svm_probability": float(result["svm_probability"][0]),
                      "routing_probability": float(result["score"][0]),
                      "policy": bundle["policies"][variant][args.mode],
                      "model": str(result["model"][0]),
                      "note": "Xác suất nhãn định tuyến; chưa gọi LLM để trả lời."}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
