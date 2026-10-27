from huggingface_hub import HfApi, create_repo

api = HfApi()

# Create the destination repository
create_repo(
    repo_id="mohjkhan/babyshark",
    repo_type="model",
    exist_ok=True,
)

sources = {
    "cra-soft": "mohjkhan/babyshark-cra-soft",
    "cra-topk": "mohjkhan/babyshark-cra-topk",
    "cra-uniform": "mohjkhan/babyshark-cra-uniform",
    "lora-baseline": "mohjkhan/babyshark-lora-baseline",
    "cra-random": "mohjkhan/babyshark-cra-random",
}

for folder, source_repo in sources.items():

    print(f"\nCopying {source_repo} -> babyshark/{folder}/")

    api.copy_files(
        source=f"hf://{source_repo}/",
        destination=f"hf://mohjkhan/babyshark/{folder}/",
    )

    print(f"✓ {folder} complete")

print("\nAll variants copied.")
