import torch


def main() -> None:
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        memory_gb = properties.total_memory / (1024**3)

        print(f"GPU: {properties.name}")
        print(f"VRAM: {memory_gb:.1f} GB")
    else:
        print("GPU training cannot start because CUDA is unavailable.")


if __name__ == "__main__":
    main()
