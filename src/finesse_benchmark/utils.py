import os
import hashlib
import json
from huggingface_hub import snapshot_download

def get_model_hash(model_path: str) -> str:
    """
    Computes a SHA256 hash of the model's contents from its local directory or file path.
    For HF remote models, resolves to local cache using snapshot_download.
    """
    original_path = model_path
    
    if not os.path.exists(model_path):
        try:
            # Assume it's a HF repo ID and resolve to cache
            cached_path = snapshot_download(model_path, cache_dir=None)  # Uses default HF cache
            model_path = cached_path
            print(f"Resolved HF model '{original_path}' to cache: {model_path}")
        except Exception as e:
            raise ValueError(f"Could not resolve HF model '{original_path}' to local path: {e}")
    
    pre_hash = hashlib.sha256()
    
    if os.path.isfile(model_path):
        with open(model_path, 'rb') as f:
            pre_hash.update(f.read())
        return pre_hash.hexdigest()
    elif os.path.isdir(model_path):
        # Walk the directory in sorted order for deterministic hashing
        for root, dirs, files in sorted(os.walk(model_path)):
            for file in sorted(files):
                if file.endswith(('.bin', '.safetensors', '.model', '.onnx', '.pt')):  # Focus on model weights
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'rb') as f:
                            pre_hash.update(f.read())
                    except Exception as e:
                        print(f"Warning: Could not hash {file_path}: {e}")
        return pre_hash.hexdigest()
    else:
        raise ValueError(f"Model path must be a file or directory: {model_path}")

def _deep_convert_to_str(obj):
    """
    Recursively convert all leaf values in a dict or list to strings,
    leaving dicts and lists intact but processing their contents.
    """
    if isinstance(obj, dict):
        return {k: _deep_convert_to_str(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_deep_convert_to_str(item) for item in obj]
    else:
        # For primitives (int, float, bool, None, etc.), convert to str
        return str(obj)

def _flatten_dict(obj, parent_key='', sep='.'):
    """
    Recursively flatten a nested dictionary into a list of 'key.path:value' strings.
    Values are converted to str. Handles only dicts and primitives (no lists).
    """
    if isinstance(obj, dict):
        items = []
        for k, v in obj.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            items.extend(_flatten_dict(v, new_key, sep))
        return items
    else:
        return [f"{parent_key}:{str(obj)}"]

def get_content_hash(data: dict, debug_file_path: str = None) -> str:
    """
    Computes a SHA256 hash using Hash Flattening: flattens the dict to key.path:value strings,
    sorts them alphabetically, joins into a single string, and hashes it.
    Processes the input data as-is (including 'content_hash': '' for fixed frame consistency).
    If debug_file_path is provided, writes the canonical_str to it for debugging.
    This ensures absolute determinism without JSON serialization dependencies.
    """
    # Flatten the data (assumes dict, no lists)
    flattened = _flatten_dict(data)
    
    # Sort alphabetically
    flattened.sort()
    
    # Join into canonical string
    canonical_str = ''.join(flattened)
    
    # Write to debug file if provided
    if debug_file_path:
        os.makedirs(os.path.dirname(debug_file_path), exist_ok=True)
        with open(debug_file_path, 'w', encoding='utf-8') as f:
            f.write(canonical_str)
    
    pre_hash = hashlib.sha256()
    pre_hash.update(canonical_str.encode('utf-8'))
    return pre_hash.hexdigest()