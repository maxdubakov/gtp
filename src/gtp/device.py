import torch

from gtp.stage2.config import DeviceConfig


def auto_device() -> str:
    if torch.backends.mps.is_available():
        return 'mps'
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def get_device_info(device: str) -> DeviceConfig:
    info = DeviceConfig(type=device)
    if device != 'cuda':
        return info
    try:
        if torch.cuda.is_available():
            info.cuda_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info.cuda_memory_gib = round(props.total_memory / (1024**3), 2)
    except Exception:
        pass
    return info
