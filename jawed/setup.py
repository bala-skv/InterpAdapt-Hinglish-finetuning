import huggingface_hub
from huggingface_hub import HfApi

print("Version:", huggingface_hub.__version__)
print("Has copy_files:", hasattr(HfApi, "copy_files"))
