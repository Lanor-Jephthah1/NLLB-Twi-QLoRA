from __future__ import annotations

import torch


def main() -> None:
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        device_index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device_index)
        total_gb = props.total_memory / 1024**3
        print(f"GPU: {torch.cuda.get_device_name(device_index)}")
        print(f"VRAM: {total_gb:.1f} GB")
        print(f"CUDA runtime: {torch.version.cuda}")


if __name__ == "__main__":
    main()
