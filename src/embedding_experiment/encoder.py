"""Encoder ONNX đóng băng: phủ toàn bộ token, mean pooling rồi chuẩn hóa L2.

Không fit trên RouterBench. Mỗi chunk có tối đa 254 content tokens + CLS/SEP.
Mean pooling theo attention mask trong chunk (gồm CLS/SEP), gộp các vector thô
với trọng số số content tokens; cuối cùng chuẩn hóa L2 một lần cho cả prompt.
"""

import json
from time import perf_counter

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from src.embedding_experiment.common import (
    DIMENSION, ENCODER_DIR, MAX_TOKENS, MODEL_ID, REVISION, sha256, write_json,
)


def split_token_ids(ids, content_limit=MAX_TOKENS - 2):
    if content_limit < 1 or not ids:
        raise ValueError("Cần token đầu vào và content_limit > 0")
    return [ids[start:start + content_limit] for start in range(0, len(ids), content_limit)]


def masked_mean(token_vectors, attention_mask):
    weights = attention_mask[:, :, None].astype(np.float32)
    return (token_vectors * weights).sum(axis=1) / weights.sum(axis=1).clip(min=1)


def normalize_rows(vectors):
    lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
    if not np.isfinite(vectors).all() or (lengths <= 0).any():
        raise ValueError("Vector embedding rỗng hoặc không hữu hạn")
    return (vectors / lengths).astype(np.float32)


class FrozenEncoder:
    def __init__(self, threads=4):
        self.manifest = json.loads((ENCODER_DIR / "manifest.json").read_text())
        if self.manifest["revision"] != REVISION or self.manifest["model_id"] != MODEL_ID:
            raise ValueError("Encoder không khớp revision đã định")
        for filename, expected in self.manifest["files"].items():
            if sha256(ENCODER_DIR / filename) != expected:
                raise ValueError(f"File encoder đã thay đổi: {filename}")
        self.tokenizer = Tokenizer.from_file(str(ENCODER_DIR / "tokenizer.json"))
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        self.cls = self.tokenizer.token_to_id("[CLS]")
        self.sep = self.tokenizer.token_to_id("[SEP]")
        self.pad = self.tokenizer.token_to_id("[PAD]")
        if any(value is None for value in [self.cls, self.sep, self.pad]):
            raise ValueError("Tokenizer thiếu special token")
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(ENCODER_DIR / "onnx/model.onnx"), options,
                                            providers=["CPUExecutionProvider"])
        self.input_names = [item.name for item in self.session.get_inputs()]
        self.threads = threads

    def encode(self, prompts, batch_size=32, progress=True):
        prompts = list(prompts)
        if not prompts or any(not isinstance(p, str) or not p.strip() for p in prompts):
            raise ValueError("Cần prompt không rỗng")
        if batch_size < 1:
            raise ValueError("batch_size phải dương")
        started = perf_counter()
        encodings = self.tokenizer.encode_batch(prompts, add_special_tokens=False)
        lengths = [len(item.ids) for item in encodings]
        chunks = [(owner, ids) for owner, item in enumerate(encodings)
                  for ids in split_token_ids(item.ids)]
        # Gom độ dài gần nhau để giảm padding; kết quả được trả về đúng thứ tự prompt.
        chunks.sort(key=lambda pair: len(pair[1]))
        sums = np.zeros((len(prompts), DIMENSION), dtype=np.float64)
        totals = np.zeros(len(prompts), dtype=np.float64)
        last_log = perf_counter()
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]
            width = max(len(ids) for _, ids in batch) + 2
            input_ids = np.full((len(batch), width), self.pad, dtype=np.int64)
            mask = np.zeros_like(input_ids)
            for row, (_, ids) in enumerate(batch):
                sequence = [self.cls, *ids, self.sep]
                input_ids[row, :len(sequence)] = sequence
                mask[row, :len(sequence)] = 1
            inputs = {"input_ids": input_ids, "attention_mask": mask,
                      "token_type_ids": np.zeros_like(input_ids)}
            output = self.session.run(None, {name: inputs[name] for name in self.input_names})[0]
            if output.shape != (len(batch), width, DIMENSION):
                raise ValueError(f"ONNX output không phải token embeddings: {output.shape}")
            pooled = masked_mean(output, mask)
            for row, (owner, ids) in enumerate(batch):
                sums[owner] += len(ids) * pooled[row]
                totals[owner] += len(ids)
            if progress and (perf_counter() - last_log >= 20 or start + batch_size >= len(chunks)):
                print(f"  Embedding: {min(start + batch_size, len(chunks)):,}/{len(chunks):,} chunks", flush=True)
                last_log = perf_counter()
        vectors = normalize_rows(sums / totals[:, None])
        return vectors, {"prompts": len(prompts), "chunks": len(chunks),
                         "multi_chunk_prompts": int(sum(n > MAX_TOKENS - 2 for n in lengths)),
                         "max_content_tokens": max(lengths), "truncated_content_tokens": 0,
                         "seconds": perf_counter() - started,
                         "batch_size": batch_size, "cpu_threads": self.threads}


def cached_embeddings(encoder, data, cache_dir, split):
    """Cache có kiểm tra nội dung/thứ tự prompt, encoder, cách gộp và SHA256 array."""
    import hashlib
    payload = list(zip(data.sample_id.tolist(), data.prompt_text.tolist()))
    signature = {
        "rows_sha256": hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode()).hexdigest(),
        "encoder": encoder.manifest, "max_tokens": MAX_TOKENS,
        "pooling": "masked_mean_per_chunk_content_weighted_mean_then_l2_v1",
        "encoder_source_sha256": sha256(__file__),
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    array_path, metadata_path = cache_dir / f"{split}.npy", cache_dir / f"{split}.json"
    if array_path.exists() and metadata_path.exists():
        record = json.loads(metadata_path.read_text())
        if record["signature"] != signature or record["array_sha256"] != sha256(array_path):
            raise ValueError(f"Cache {split} không khớp dữ liệu/encoder")
        array = np.load(array_path, allow_pickle=False)
        if array.shape != (len(data), DIMENSION) or not np.isfinite(array).all():
            raise ValueError("Cache embedding không hợp lệ")
        return array, record
    array, stats = encoder.encode(data.prompt_text)
    np.save(array_path, array, allow_pickle=False)
    record = {"signature": signature, "array_sha256": sha256(array_path), "stats": stats}
    write_json(metadata_path, record)
    return array, record
