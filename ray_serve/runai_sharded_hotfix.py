import glob
import os
import time

import torch

_HOTFIX_APPLIED = False


def patch_runai_dtype() -> None:
    import runai_model_streamer.safetensors_streamer.safetensors_pytorch as sp

    sp.safetenors_to_torch_dtype["F8_E8M0"] = torch.uint8


def patch_marlin_load_order() -> None:
    from vllm.model_executor.model_loader import base_loader
    from vllm.model_executor.model_loader.utils import (
        initialize_model,
        process_weights_after_loading,
    )
    from vllm.utils.torch_utils import set_default_torch_dtype

    def load_model(self, vllm_config, model_config):
        device_config = vllm_config.device_config
        load_config = vllm_config.load_config
        load_device = device_config.device if load_config.device is None else load_config.device
        target_device = torch.device(load_device)
        with set_default_torch_dtype(model_config.dtype):
            with target_device:
                model = initialize_model(vllm_config=vllm_config, model_config=model_config)
            if load_config.load_format == "runai_streamer_sharded":
                process_weights_after_loading(model, model_config, target_device)
            self.load_weights(model, model_config)
            if load_config.load_format != "runai_streamer_sharded":
                process_weights_after_loading(model, model_config, target_device)
        return model.eval()

    base_loader.BaseModelLoader.load_model = load_model


_ATTN_SCALE_SUFFIXES = ("_k_scale", "_v_scale", "_q_scale", "_prob_scale")
_ATTN_SCALE_TO_FLOAT = {
    "_k_scale": "_k_scale_float",
    "_v_scale": "_v_scale_float",
    "_q_scale": "_q_scale_float",
    "_prob_scale": "_prob_scale_float",
}


def is_attention_scale_key(key: str) -> bool:
    return "attn.attn." in key and key.endswith(_ATTN_SCALE_SUFFIXES)


def apply_attention_scale(model, key: str, tensor: torch.Tensor) -> bool:
    parts = key.rsplit(".", 1)
    if len(parts) != 2:
        return False
    submodule_path, attr_name = parts
    try:
        submodule = model.get_submodule(submodule_path)
    except Exception:
        try:
            submodule = model.get_submodule(f"model.{submodule_path}")
        except Exception:
            return False
    if tensor.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        tensor = tensor.float()
    device = next(model.parameters()).device
    t = tensor.to(device=device)
    if hasattr(submodule, attr_name):
        existing = getattr(submodule, attr_name)
        if hasattr(existing, "data"):
            if existing.shape != t.shape:
                if existing.numel() != t.numel():
                    return False
                t = t.reshape_as(existing)
            existing.data.copy_(t.to(dtype=existing.dtype))
        else:
            setattr(submodule, attr_name, t.detach().clone())
    else:
        setattr(submodule, attr_name, t.detach().clone())
    float_attr = _ATTN_SCALE_TO_FLOAT.get(attr_name)
    if float_attr and hasattr(submodule, float_attr):
        setattr(submodule, float_attr, t.item() if t.numel() == 1 else float(t.float().mean().item()))
    return True


def copy_tensor_into_param(state_dict: dict[str, torch.Tensor], key: str, tensor: torch.Tensor) -> bool:
    if key not in state_dict:
        return False
    param = state_dict[key]
    if tensor.dtype == torch.uint8 and param.dtype == torch.float8_e8m0fnu:
        tensor = tensor.view(torch.float8_e8m0fnu)
    if tensor.shape != param.shape and tensor.numel() == param.numel():
        if (
            tensor.dim() >= 2
            and tensor.shape[:-2] == param.shape[:-2]
            and tensor.shape[-2] == param.shape[-1]
            and tensor.shape[-1] == param.shape[-2]
        ):
            tensor = tensor.transpose(-2, -1).contiguous()
        else:
            tensor = tensor.reshape(param.shape)
    if tensor.shape != param.shape:
        if tensor.numel() != param.numel():
            return False
        tensor = tensor.reshape(param.shape)
    param_data = param.data
    for dim, size in enumerate(tensor.shape):
        if size < param.shape[dim]:
            param_data = param_data.narrow(dim, 0, size)
    param_data.copy_(tensor)
    state_dict.pop(key)
    return True


def patch_sharded_loader() -> None:
    from vllm.distributed import get_tensor_model_parallel_rank
    from vllm.model_executor.model_loader import sharded_state_loader as ssl
    from vllm.transformers_utils.s3_utils import glob as s3_glob
    from vllm.transformers_utils.utils import is_s3

    def load_weights(self, model, model_config):
        model_weights = model_config.model_weights if hasattr(model_config, "model_weights") else model_config.model
        rank = get_tensor_model_parallel_rank()
        pattern = os.path.join(model_weights, self.pattern.format(rank=rank, part="*"))
        if is_s3(model_weights):
            file_pattern = f"*{self.pattern.format(rank=rank, part='*')}"
            filepaths = s3_glob(path=model_weights, allow_pattern=[file_pattern])
        else:
            filepaths = glob.glob(pattern)
        if not filepaths:
            raise ValueError(f"Could not find checkpoint files '{pattern}'")
        state_dict = ssl.ShardedStateLoader._filter_subtensors(model.state_dict())
        mapper = getattr(type(model), "hf_to_vllm_mapper", None) or getattr(model, "hf_to_vllm_mapper", None)
        deferred_attn_scales: list[tuple[str, torch.Tensor]] = []
        t0 = time.perf_counter()
        for key, tensor in self.iterate_over_files(filepaths):
            mapped_key = mapper._map_name(key) if mapper is not None and hasattr(mapper, "_map_name") else key
            mapped_key = key if mapped_key is None else mapped_key
            if copy_tensor_into_param(state_dict, mapped_key, tensor):
                continue
            if copy_tensor_into_param(state_dict, key, tensor):
                continue
            if is_attention_scale_key(mapped_key):
                deferred_attn_scales.append((mapped_key, tensor.detach().clone()))
            elif is_attention_scale_key(key):
                deferred_attn_scales.append((key, tensor.detach().clone()))
        for scale_key, scale_tensor in deferred_attn_scales:
            apply_attention_scale(model, scale_key, scale_tensor)
        ssl.logger.info_once("Loading weights took %.2f seconds", time.perf_counter() - t0, scope="local")
        if state_dict:
            raise ValueError(f"Missing keys {tuple(state_dict)} in loaded state!")

    ssl.ShardedStateLoader.load_weights = load_weights


def apply_hotfixes() -> None:
    global _HOTFIX_APPLIED
    if _HOTFIX_APPLIED:
        return
    patch_runai_dtype()
    patch_marlin_load_order()
    patch_sharded_loader()
    _HOTFIX_APPLIED = True
