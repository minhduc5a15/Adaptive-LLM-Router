"""Tải encoder công khai đúng revision; không tải dữ liệu prompt lên mạng."""

from huggingface_hub import snapshot_download

from src.embedding_experiment.common import ENCODER_DIR, MODEL_ID, REVISION, sha256, write_json

FILES = ["onnx/model.onnx", "tokenizer.json", "tokenizer_config.json", "config.json",
         "sentence_bert_config.json", "1_Pooling/config.json", "modules.json", "README.md"]


def main():
    ENCODER_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(MODEL_ID, revision=REVISION, allow_patterns=FILES,
                      local_dir=ENCODER_DIR, token=False)
    record = {"model_id": MODEL_ID, "revision": REVISION,
              "files": {name: sha256(ENCODER_DIR / name) for name in FILES}}
    write_json(ENCODER_DIR / "manifest.json", record)
    print(f"Đã tải encoder: {ENCODER_DIR}")
    print(f"Revision: {REVISION}")


if __name__ == "__main__":
    main()
