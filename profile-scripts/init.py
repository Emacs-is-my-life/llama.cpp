import gguf
import gc
from typing import Dict


def get_aligned_n_bytes(n_bytes: int, align_bytes: int) -> int:
    return ((n_bytes + align_bytes - 1) // align_bytes) * align_bytes


def gguf_get_weight_tensor_size_table(gguf_path: str) -> Dict[str, int]:
    reader = gguf.GGUFReader(gguf_path)

    tensor_size_table = {}
    for tensor in reader.tensors:
        aligned_n_bytes = get_aligned_n_bytes(tensor.n_bytes, 64)
        tensor_size_table[f"<x>{tensor.name}"] = aligned_n_bytes

    del reader
    gc.collect()
    return tensor_size_table


