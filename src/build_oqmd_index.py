from chemistry.oqmd_local import build_local_oqmd_index


def main() -> None:
    print("Building the local OQMD fallback index...")
    path = build_local_oqmd_index(force=True)
    print(f"Local OQMD index ready: {path}")


if __name__ == "__main__":
    main()
