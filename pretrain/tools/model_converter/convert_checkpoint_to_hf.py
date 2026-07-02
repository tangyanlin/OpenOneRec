"""Checkpoint to HuggingFace Format Converter

This module provides utilities to convert PyTorch checkpoints (DCP or .pth files)
to HuggingFace format (safetensors or bin files with sharding support).
"""

import argparse
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Dict, Optional, Union

import torch
import tqdm
from safetensors.torch import save_file
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.default_planner import _EmptyStateDictLoadPlanner
from torch.distributed.checkpoint.metadata import STATE_DICT_TYPE
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Constants
SHARD_FNAME_TEMPLATE = "model-{cpt_idx}-of-{num_shards}"
BYTES_PER_GB = 1024 * 1024 * 1024
DEFAULT_MAX_GB_PER_SHARD = 5
DEFAULT_DTYPE = "bf16"

# Common HuggingFace config files to copy
HF_CONFIG_FILES = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",  # SentencePiece tokenizer model file
    "vocab.txt",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
    "generation_config.json",
    "preprocessor_config.json",  # For vision models
]


def _get_torch_dtype(dtype_str: str) -> torch.dtype:
    """Convert dtype string to torch.dtype.
    
    Args:
        dtype_str: Data type string ("fp32", "fp16", "bf16")
        
    Returns:
        Corresponding torch.dtype
        
    Raises:
        ValueError: If dtype_str is not supported
    """
    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    if dtype_str not in dtype_map:
        raise ValueError(f"Unsupported dtype: {dtype_str}. Supported: {list(dtype_map.keys())}")
    return dtype_map[dtype_str]


def _extract_state_dict_from_checkpoint(checkpoint: Dict, model_only: bool = True) -> Dict[str, torch.Tensor]:
    """Extract state_dict from checkpoint with various structures.
    
    Args:
        checkpoint: Checkpoint dictionary
        model_only: Whether to extract only model weights
        
    Returns:
        State dictionary containing model weights
    """
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported checkpoint format: {type(checkpoint)}")
    
    # Check for nested DCP-like structure
    if model_only and "app" in checkpoint and "model" in checkpoint["app"]:
        logger.info("Found nested structure: checkpoint['app']['model']")
        return checkpoint["app"]["model"]
    elif "model" in checkpoint:
        logger.info("Found structure: checkpoint['model']")
        return checkpoint["model"]
    elif "state_dict" in checkpoint:
        logger.info("Found structure: checkpoint['state_dict']")
        return checkpoint["state_dict"]
    else:
        # Assume entire dict is the state_dict
        logger.info("Using entire checkpoint as state_dict")
        return checkpoint


def _convert_state_dict_to_shards(
    state_dict: Dict[str, torch.Tensor],
    output_dir: Union[str, os.PathLike],
    use_safetensor: bool = True,
    max_gb_per_shard: int = DEFAULT_MAX_GB_PER_SHARD,
    dtype: str = DEFAULT_DTYPE
) -> None:
    """Convert state_dict to sharded safetensors or bin files.
    
    Args:
        state_dict: State dictionary containing model weights
        output_dir: Output directory for sharded files
        use_safetensor: Whether to use safetensors format (default: True)
        max_gb_per_shard: Maximum size per shard in GB (default: 5)
        dtype: Data type for conversion ("fp32", "fp16", "bf16", default: "bf16")
        
    Raises:
        ValueError: If dtype is not supported
    """
    torch_dtype = _get_torch_dtype(dtype)
    logger.info(f"Converting state_dict to {dtype} format")
    
    # Convert data types
    logger.info("Converting tensor data types...")
    for key in tqdm.tqdm(state_dict.keys(), desc="Converting dtypes"):
        state_dict[key] = state_dict[key].to(torch_dtype)
    
    # Split into shards
    logger.info(f"Splitting state_dict into shards (max {max_gb_per_shard} GB per shard)...")
    split_state_dicts: Dict[int, Dict[str, torch.Tensor]] = {}
    shard_idx = 0
    total_size = 0
    current_size = 0
    
    max_bytes_per_shard = max_gb_per_shard * BYTES_PER_GB
    
    for key, weight in tqdm.tqdm(state_dict.items(), desc="Creating shards"):
        if shard_idx not in split_state_dicts:
            split_state_dicts[shard_idx] = {}
        
        split_state_dicts[shard_idx][key] = weight
        weight_size = weight.numel() * weight.element_size()
        current_size += weight_size
        total_size += weight_size
        
        if current_size >= max_bytes_per_shard:
            shard_idx += 1
            current_size = 0
    
    # Write shard files
    num_shards = len(split_state_dicts)
    weight_map: Dict[str, str] = {}
    output_path_obj = Path(output_dir)
    output_path_obj.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Writing {num_shards} shard files...")
    for shard_idx, shard_state_dict in tqdm.tqdm(split_state_dicts.items(), desc="Writing shards"):
        shard_name = SHARD_FNAME_TEMPLATE.format(
            cpt_idx=f"{shard_idx}".zfill(5),
            num_shards=f"{num_shards}".zfill(5)
        )
        
        if use_safetensor:
            shard_path = output_path_obj / f"{shard_name}.safetensors"
            save_file(shard_state_dict, shard_path, metadata={"format": "pt"})
        else:
            shard_path = output_path_obj / f"{shard_name}.bin"
            torch.save(shard_state_dict, shard_path)
        
        # Update weight map
        shard_filename = shard_path.name
        for key in shard_state_dict.keys():
            weight_map[key] = shard_filename
        
        shard_size_gb = os.path.getsize(shard_path) / BYTES_PER_GB
        logger.info(f"Shard {shard_idx + 1}/{num_shards}: {shard_size_gb:.2f} GiB saved to {shard_path}")
    
    # Write index file
    index_filename = "model.safetensors.index.json" if use_safetensor else "model.bin.index.json"
    index_path = output_path_obj / index_filename
    
    index_data = {
        "metadata": {
            "total_size": total_size
        },
        "weight_map": weight_map,
    }
    
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index_data, f, indent=2)
    
    logger.info(f"Index file saved to {index_path}")
    logger.info(f"Total model size: {total_size / BYTES_PER_GB:.2f} GiB")


def pth_to_hf_format(
    pth_file_path: Union[str, os.PathLike],
    output_dir: Union[str, os.PathLike],
    model_only: bool = True,
    use_safetensor: bool = True,
    max_gb_per_shard: int = DEFAULT_MAX_GB_PER_SHARD,
    dtype: str = DEFAULT_DTYPE
) -> None:
    """Convert .pth file to HuggingFace format (safetensors or bin files).
    
    Args:
        pth_file_path: Path to .pth checkpoint file
        output_dir: Output directory for converted files
        model_only: Whether to extract only model weights (default: True)
        use_safetensor: Whether to use safetensors format (default: True)
        max_gb_per_shard: Maximum size per shard in GB (default: 5)
        dtype: Data type for conversion (default: "bf16")
        
    Raises:
        FileNotFoundError: If pth_file_path does not exist
        ValueError: If pth_file_path is not a .pth file or has unsupported format
        
    .. warning::
        To avoid OOM, it's recommended to run this function on a single rank/process.
    """
    pth_path = Path(pth_file_path)
    
    if not pth_path.exists():
        raise FileNotFoundError(f"PTH file not found: {pth_path}")
    
    if pth_path.suffix != ".pth":
        raise ValueError(f"Expected .pth file, got: {pth_path.suffix}")
    
    logger.info(f"Loading PTH file from {pth_path}...")
    checkpoint = torch.load(pth_path, map_location="cpu")
    
    # Extract state_dict from checkpoint
    state_dict = _extract_state_dict_from_checkpoint(checkpoint, model_only=model_only)
    logger.info(f"Loaded state_dict with {len(state_dict)} keys")
    
    # Convert to HuggingFace format
    _convert_state_dict_to_shards(
        state_dict=state_dict,
        output_dir=output_dir,
        use_safetensor=use_safetensor,
        max_gb_per_shard=max_gb_per_shard,
        dtype=dtype
    )


def dcp_to_hf_format(
    dcp_checkpoint_dir: Union[str, os.PathLike],
    output_dir: Union[str, os.PathLike],
    model_only: bool = True,
    use_safetensor: bool = True,
    max_gb_per_shard: int = DEFAULT_MAX_GB_PER_SHARD,
    dtype: str = DEFAULT_DTYPE
) -> None:
    """Convert DCP (Distributed Checkpoint) to HuggingFace format.
    
    Args:
        dcp_checkpoint_dir: Directory containing the DCP checkpoint
        output_dir: Output directory for converted files
        model_only: Whether to extract only model weights (default: True)
        use_safetensor: Whether to use safetensors format (default: True)
        max_gb_per_shard: Maximum size per shard in GB (default: 5)
        dtype: Data type for conversion (default: "bf16")
        
    Raises:
        FileNotFoundError: If dcp_checkpoint_dir does not exist
        
    .. warning::
        To avoid OOM, it's recommended to run this function on a single rank/process.
    """
    dcp_path = Path(dcp_checkpoint_dir)
    
    if not dcp_path.exists():
        raise FileNotFoundError(f"DCP checkpoint directory not found: {dcp_path}")
    
    if not dcp_path.is_dir():
        raise ValueError(f"Expected directory, got: {dcp_path}")
    
    logger.info(f"Loading DCP checkpoint from {dcp_path}...")
    state_dict: STATE_DICT_TYPE = {}
    
    _load_state_dict(
        state_dict,
        storage_reader=FileSystemReader(str(dcp_path)),
        planner=_EmptyStateDictLoadPlanner(),
        no_dist=True,
    )
    
    logger.info("DCP checkpoint loaded successfully")
    
    if model_only:
        if "app" not in state_dict or "model" not in state_dict["app"]:
            raise ValueError("Expected 'app.model' in DCP checkpoint when model_only=True")
        state_dict = state_dict["app"]["model"]
        logger.info(f"Extracted model state_dict with {len(state_dict)} keys")
    
    # Convert to HuggingFace format
    _convert_state_dict_to_shards(
        state_dict=state_dict,
        output_dir=output_dir,
        use_safetensor=use_safetensor,
        max_gb_per_shard=max_gb_per_shard,
        dtype=dtype
    )


def copy_hf_config_files(
    source_hf_model_path: Union[str, os.PathLike],
    output_dir: Union[str, os.PathLike]
) -> None:
    """Copy HuggingFace configuration files from source to output directory.
    
    Args:
        source_hf_model_path: Path to source HuggingFace model directory
        output_dir: Output directory where config files will be copied
    """
    source_path = Path(source_hf_model_path)
    output_path = Path(output_dir)
    
    if not source_path.exists():
        logger.warning(f"Source HuggingFace model path does not exist: {source_path}")
        return
    
    if not source_path.is_dir():
        logger.warning(f"Source path is not a directory: {source_path}")
        return
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    copied_files = []
    
    # Copy known config files
    for config_file in HF_CONFIG_FILES:
        source_file = source_path / config_file
        if source_file.exists():
            dest_file = output_path / config_file
            shutil.copy2(source_file, dest_file)
            copied_files.append(config_file)
            logger.debug(f"Copied {config_file} to {output_path}")
    
    # Copy additional JSON and TXT files (may be config files)
    for pattern in ["*.json", "*.txt"]:
        for source_file in source_path.glob(pattern):
            # Skip already copied files and weight files
            if (source_file.name in copied_files or 
                source_file.name.startswith("model-") or
                source_file.suffix in [".bin", ".safetensors"]):
                continue
            
            dest_file = output_path / source_file.name
            if not dest_file.exists():  # Avoid overwriting already copied files
                shutil.copy2(source_file, dest_file)
                if source_file.name not in HF_CONFIG_FILES:
                    logger.debug(f"Copied additional file: {source_file.name}")
    
    if copied_files:
        logger.info(f"Successfully copied {len(copied_files)} config files from {source_path} to {output_path}")
    else:
        logger.warning(f"No config files found in {source_path}")


def _convert_seqcls_to_causal_lm_state_dict(
    state_dict: Dict[str, torch.Tensor],
    vocab_size: int,
    save_score_weight_path: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    """Convert Qwen3ForSequenceClassification state_dict to Qwen3ForCausalLM state_dict.
    
    This function:
    1. Removes the 'score.weight' key (SeqCls classification head)
    2. Adds 'lm_head.weight' initialized from embed_tokens (tied weights)
    3. Optionally saves the 'score.weight' to a file for later restoration
    
    Args:
        state_dict: State dict from Qwen3ForSequenceClassification
        vocab_size: Vocabulary size for the lm_head output dimension
        save_score_weight_path: If provided, save score.weight to this path for
            later restoration when converting back to SeqCls after distillation.
        
    Returns:
        State dict compatible with Qwen3ForCausalLM
    """
    new_state_dict = {}
    score_weight = None
    
    for key, value in state_dict.items():
        if key == "score.weight":
            score_weight = value
            logger.info(f"Extracting SeqCls head: {key} (shape={value.shape})")
            continue
        new_state_dict[key] = value
    
    # Save score.weight for later restoration
    if score_weight is not None and save_score_weight_path is not None:
        save_dir = Path(save_score_weight_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"score.weight": score_weight}, save_score_weight_path)
        logger.info(f"Saved score.weight to {save_score_weight_path} (shape={score_weight.shape})")
    
    # Add lm_head.weight from embed_tokens (tied weights)
    embed_key = "model.embed_tokens.weight"
    if embed_key in new_state_dict:
        new_state_dict["lm_head.weight"] = new_state_dict[embed_key].clone()
        logger.info(f"Added lm_head.weight from {embed_key} (shape={new_state_dict[embed_key].shape})")
    else:
        logger.warning(f"'{embed_key}' not found in state_dict, cannot create lm_head.weight")
    
    return new_state_dict


def _convert_causal_lm_to_seqcls_state_dict(
    state_dict: Dict[str, torch.Tensor],
    score_weight_path: str,
    num_labels: int = 2,
) -> Dict[str, torch.Tensor]:
    """Convert Qwen3ForCausalLM state_dict back to Qwen3ForSequenceClassification state_dict.
    
    This function:
    1. Removes 'lm_head.weight' key
    2. Restores 'score.weight' from the saved file (from original SeqCls training)
    
    After verl_distillation, the transformer backbone (model.*) weights have been
    further optimized. The score.weight is restored from the original SeqCls checkpoint
    so that click-through rate prediction can still be performed.
    
    Note: If you want to fine-tune the score head after distillation, you can
    retrain it using the SeqCls training script with the distilled model as init.
    
    Args:
        state_dict: State dict from Qwen3ForCausalLM (after distillation)
        score_weight_path: Path to the saved score.weight file
        num_labels: Number of labels for the classification head (default: 2)
        
    Returns:
        State dict compatible with Qwen3ForSequenceClassification
    """
    new_state_dict = {}
    
    # Load saved score.weight
    score_data = torch.load(score_weight_path, map_location="cpu")
    score_weight = score_data["score.weight"]
    logger.info(f"Loaded score.weight from {score_weight_path} (shape={score_weight.shape})")
    
    for key, value in state_dict.items():
        if key == "lm_head.weight":
            logger.info(f"Removing CausalLM head: {key} (shape={value.shape})")
            continue
        new_state_dict[key] = value
    
    # Restore score.weight
    new_state_dict["score.weight"] = score_weight
    logger.info(f"Restored score.weight (shape={score_weight.shape})")
    
    return new_state_dict


def get_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser.
    
    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="Convert PyTorch checkpoints (DCP or .pth) to HuggingFace format"
    )
    
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to DCP checkpoint directory or .pth file"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for converted HuggingFace model"
    )
    
    parser.add_argument(
        "--source_hf_model_path",
        type=str,
        default=None,
        help="Path to original HuggingFace model to copy config files from (optional)"
    )
    
    parser.add_argument(
        "--use_safetensor",
        action="store_true",
        default=True,
        help="Use safetensors format (default: True)"
    )
    
    parser.add_argument(
        "--no_safetensor",
        dest="use_safetensor",
        action="store_false",
        help="Use .bin format instead of safetensors"
    )
    
    parser.add_argument(
        "--max_gb_per_shard",
        type=int,
        default=DEFAULT_MAX_GB_PER_SHARD,
        help=f"Maximum size per shard in GB (default: {DEFAULT_MAX_GB_PER_SHARD})"
    )
    
    parser.add_argument(
        "--dtype",
        type=str,
        default=DEFAULT_DTYPE,
        choices=["fp32", "fp16", "bf16"],
        help=f"Data type for conversion (default: {DEFAULT_DTYPE})"
    )
    
    parser.add_argument(
        "--convert_seqcls_to_causal_lm",
        action="store_true",
        default=False,
        help="Convert Qwen3ForSequenceClassification checkpoint to Qwen3ForCausalLM format. "
             "Removes 'score.weight', adds 'lm_head.weight' from tied embeddings, "
             "and saves score.weight for later restoration."
    )
    
    parser.add_argument(
        "--convert_causal_lm_to_seqcls",
        action="store_true",
        default=False,
        help="Convert Qwen3ForCausalLM HF model back to Qwen3ForSequenceClassification format. "
             "Removes 'lm_head.weight' and restores 'score.weight' from a saved file. "
             "Use this after verl_distillation to restore click prediction capability."
    )
    
    parser.add_argument(
        "--score_weight_path",
        type=str,
        default=None,
        help="Path to save/load the score.weight file. "
             "When used with --convert_seqcls_to_causal_lm: path to save score.weight (default: <output_dir>/score_weight.pt). "
             "When used with --convert_causal_lm_to_seqcls: path to load score.weight (required)."
    )
    
    parser.add_argument(
        "--num_labels",
        type=int,
        default=2,
        help="Number of labels for SequenceClassification (default: 2, for binary click prediction)"
    )
    
    return parser


def main() -> None:
    """Main entry point for the script."""
    parser = get_argument_parser()
    args = parser.parse_args()
    
    checkpoint_path = Path(args.checkpoint_dir)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")
    
    # Auto-detect input type: .pth file or DCP checkpoint directory
    if checkpoint_path.is_file() and checkpoint_path.suffix == ".pth":
        logger.info(f"Detected PTH file: {checkpoint_path}")
        pth_to_hf_format(
            pth_file_path=checkpoint_path,
            output_dir=args.output_dir,
            model_only=True,
            use_safetensor=args.use_safetensor,
            max_gb_per_shard=args.max_gb_per_shard,
            dtype=args.dtype
        )
    elif checkpoint_path.is_dir():
        logger.info(f"Detected DCP checkpoint directory: {checkpoint_path}")
        dcp_to_hf_format(
            dcp_checkpoint_dir=checkpoint_path,
            output_dir=args.output_dir,
            model_only=True,
            use_safetensor=args.use_safetensor,
            max_gb_per_shard=args.max_gb_per_shard,
            dtype=args.dtype
        )
    else:
        raise ValueError(
            f"Invalid checkpoint path: {checkpoint_path}. "
            "Expected either a .pth file or a DCP checkpoint directory."
        )
    
    # Convert SeqCls to CausalLM if requested
    if args.convert_seqcls_to_causal_lm:
        logger.info("Converting Qwen3ForSequenceClassification to Qwen3ForCausalLM format...")
        # We need to load the already-converted state_dict, transform it, and re-save
        # This happens after the initial conversion, so we read the safetensors files
        import glob as _glob
        from safetensors.torch import load_file as safe_load_file
        
        output_path_obj = Path(args.output_dir)
        
        # Determine score.weight save path
        score_weight_save_path = args.score_weight_path
        if score_weight_save_path is None:
            score_weight_save_path = str(output_path_obj / "score_weight.pt")
        logger.info(f"score.weight will be saved to: {score_weight_save_path}")
        
        # Load all safetensors shards
        combined_state_dict = {}
        shard_files = sorted(_glob.glob(str(output_path_obj / "model-*.safetensors")))
        if not shard_files:
            # Try .bin files
            shard_files = sorted(_glob.glob(str(output_path_obj / "model-*.bin")))
            for shard_file in shard_files:
                shard_data = torch.load(shard_file, map_location="cpu")
                combined_state_dict.update(shard_data)
        else:
            for shard_file in shard_files:
                shard_data = safe_load_file(shard_file)
                combined_state_dict.update(shard_data)
        
        # Convert state dict (saves score.weight for later restoration)
        vocab_size = combined_state_dict.get("model.embed_tokens.weight", torch.zeros(1)).shape[0]
        converted_state_dict = _convert_seqcls_to_causal_lm_state_dict(
            combined_state_dict, vocab_size=vocab_size,
            save_score_weight_path=score_weight_save_path,
        )
        
        # Remove old shard files
        for shard_file in shard_files:
            os.remove(shard_file)
            logger.info(f"Removed old shard: {shard_file}")
        
        # Remove old index file
        for idx_file in _glob.glob(str(output_path_obj / "model.safetensors.index.json")) + \
                        _glob.glob(str(output_path_obj / "model.bin.index.json")):
            os.remove(idx_file)
            logger.info(f"Removed old index: {idx_file}")
        
        # Re-save converted state dict
        _convert_state_dict_to_shards(
            state_dict=converted_state_dict,
            output_dir=args.output_dir,
            use_safetensor=args.use_safetensor,
            max_gb_per_shard=args.max_gb_per_shard,
            dtype=args.dtype
        )
        
        # Update config.json to use Qwen3ForCausalLM architecture
        config_path = output_path_obj / "config.json"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = json.load(f)
            if "architectures" in config_data:
                old_arch = config_data["architectures"]
                config_data["architectures"] = ["Qwen3ForCausalLM"]
                # Remove num_labels if present (not needed for CausalLM)
                config_data.pop("num_labels", None)
                # Ensure id2label and label2id are removed
                config_data.pop("id2label", None)
                config_data.pop("label2id", None)
                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump(config_data, f, indent=2, ensure_ascii=False)
                logger.info(f"Updated config.json: architectures {old_arch} -> ['Qwen3ForCausalLM']")
    
    # Convert CausalLM to SeqCls if requested (for restoring click prediction after distillation)
    if args.convert_causal_lm_to_seqcls:
        logger.info("Converting Qwen3ForCausalLM back to Qwen3ForSequenceClassification format...")
        import glob as _glob
        from safetensors.torch import load_file as safe_load_file
        
        if args.score_weight_path is None:
            raise ValueError(
                "--score_weight_path is required when using --convert_causal_lm_to_seqcls. "
                "Provide the path to the score_weight.pt file saved during SeqCls→CausalLM conversion."
            )
        
        output_path_obj = Path(args.output_dir)
        
        # Load all safetensors shards from the CausalLM model
        combined_state_dict = {}
        shard_files = sorted(_glob.glob(str(output_path_obj / "model-*.safetensors")))
        if not shard_files:
            shard_files = sorted(_glob.glob(str(output_path_obj / "model-*.bin")))
            for shard_file in shard_files:
                shard_data = torch.load(shard_file, map_location="cpu")
                combined_state_dict.update(shard_data)
        else:
            for shard_file in shard_files:
                shard_data = safe_load_file(shard_file)
                combined_state_dict.update(shard_data)
        
        # Convert state dict: remove lm_head, restore score.weight
        converted_state_dict = _convert_causal_lm_to_seqcls_state_dict(
            combined_state_dict,
            score_weight_path=args.score_weight_path,
            num_labels=args.num_labels,
        )
        
        # Remove old shard files
        for shard_file in shard_files:
            os.remove(shard_file)
            logger.info(f"Removed old shard: {shard_file}")
        
        # Remove old index file
        for idx_file in _glob.glob(str(output_path_obj / "model.safetensors.index.json")) + \
                        _glob.glob(str(output_path_obj / "model.bin.index.json")):
            os.remove(idx_file)
            logger.info(f"Removed old index: {idx_file}")
        
        # Re-save converted state dict
        _convert_state_dict_to_shards(
            state_dict=converted_state_dict,
            output_dir=args.output_dir,
            use_safetensor=args.use_safetensor,
            max_gb_per_shard=args.max_gb_per_shard,
            dtype=args.dtype
        )
        
        # Update config.json to use Qwen3ForSequenceClassification architecture
        config_path = output_path_obj / "config.json"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = json.load(f)
            old_arch = config_data.get("architectures", [])
            config_data["architectures"] = ["Qwen3ForSequenceClassification"]
            config_data["num_labels"] = args.num_labels
            config_data["id2label"] = {str(i): f"LABEL_{i}" for i in range(args.num_labels)}
            config_data["label2id"] = {f"LABEL_{i}": i for i in range(args.num_labels)}
            # Ensure pad_token_id is set (required for SeqCls batched inference)
            if "pad_token_id" not in config_data or config_data["pad_token_id"] is None:
                config_data["pad_token_id"] = 151643
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Updated config.json: architectures {old_arch} -> ['Qwen3ForSequenceClassification'], num_labels={args.num_labels}")
    
    # Copy config files if source model path is provided
    if args.source_hf_model_path:
        logger.info(f"Copying config files from {args.source_hf_model_path} to {args.output_dir}")
        copy_hf_config_files(
            source_hf_model_path=args.source_hf_model_path,
            output_dir=args.output_dir
        )
        
        # If converting SeqCls to CausalLM, also fix the copied config.json
        if args.convert_seqcls_to_causal_lm:
            config_path = Path(args.output_dir) / "config.json"
            if config_path.exists():
                with open(config_path, "r", encoding="utf-8") as f:
                    config_data = json.load(f)
                if "architectures" in config_data and config_data["architectures"] != ["Qwen3ForCausalLM"]:
                    old_arch = config_data["architectures"]
                    config_data["architectures"] = ["Qwen3ForCausalLM"]
                    config_data.pop("num_labels", None)
                    config_data.pop("id2label", None)
                    config_data.pop("label2id", None)
                    with open(config_path, "w", encoding="utf-8") as f:
                        json.dump(config_data, f, indent=2, ensure_ascii=False)
                    logger.info(f"Fixed copied config.json: architectures {old_arch} -> ['Qwen3ForCausalLM']")
        
        # If converting CausalLM to SeqCls, also fix the copied config.json
        if args.convert_causal_lm_to_seqcls:
            config_path = Path(args.output_dir) / "config.json"
            if config_path.exists():
                with open(config_path, "r", encoding="utf-8") as f:
                    config_data = json.load(f)
                old_arch = config_data.get("architectures", [])
                config_data["architectures"] = ["Qwen3ForSequenceClassification"]
                config_data["num_labels"] = args.num_labels
                config_data["id2label"] = {str(i): f"LABEL_{i}" for i in range(args.num_labels)}
                config_data["label2id"] = {f"LABEL_{i}": i for i in range(args.num_labels)}
                if "pad_token_id" not in config_data or config_data["pad_token_id"] is None:
                    config_data["pad_token_id"] = 151643
                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump(config_data, f, indent=2, ensure_ascii=False)
                logger.info(f"Fixed copied config.json: architectures {old_arch} -> ['Qwen3ForSequenceClassification']")
    
    logger.info("Conversion completed successfully!")


if __name__ == "__main__":
    main()
