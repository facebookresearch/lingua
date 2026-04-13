import argparse
import os
from typing import Optional

from requests.exceptions import HTTPError

TOKENIZER = {
    "llama2": ("meta-llama/Llama-2-7b", "tokenizer.model"),
    "llama3": ("meta-llama/Meta-Llama-3-8B", "original/tokenizer.model"),
    "gemma": ("google/gemma-2-9b", "tokenizer.model"),
}


def main(tokenizer_name: str, path_to_save: str, api_key: Optional[str] = None):
    if tokenizer_name in TOKENIZER:
        repo_id, filename = TOKENIZER[tokenizer_name]

        from huggingface_hub import hf_hub_download

        try:
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=path_to_save,
                local_dir_use_symlinks=False,
                token=api_key if api_key else None,
            )
        except HTTPError as e:
            if e.response.status_code == 401:
                print(
                    "You need to pass a valid `--hf_token=...` to download private checkpoints."
                )
            else:
                raise e
    else:
        from tiktoken import get_encoding
        if "TIKTOKEN_CACHE_DIR" not in os.environ:
            os.environ["TIKTOKEN_CACHE_DIR"] = path_to_save
        try:
            get_encoding(tokenizer_name)
        except ValueError:
            print(
                f"Tokenizer {tokenizer_name} not found. Please check the name and try again."
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("tokenizer_name", type=str)
    parser.add_argument("tokenizer_dir", type=str, default=8)
    parser.add_argument("--api_key", type=str, default="")
    args = parser.parse_args()

    main(tokenizer_name=args.tokenizer_name, path_to_save=args.tokenizer_dir, api_key=args.api_key)