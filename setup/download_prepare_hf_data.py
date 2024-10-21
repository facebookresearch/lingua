# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import subprocess
import sys
from huggingface_hub import snapshot_download


def run_command(command):
    print(f"Running: {command}")
    subprocess.run(command, shell=True, check=True)


def download_dataset(repo_id, local_dir, allow_patterns):
    print(f"Downloading dataset from {repo_id}...")
    snapshot_download(
        repo_id,
        repo_type="dataset",
        local_dir=local_dir,
        allow_patterns=allow_patterns,
        resume_download=True,
        max_workers=8,
    )
    print(f"Dataset downloaded to {local_dir}")


def parquet_to_jsonl(dataset, work_dir, src_dir, tgt_dir, ntasks=64):
    from datatrove.executor import LocalPipelineExecutor
    from datatrove.pipeline.readers import ParquetReader
    from datatrove.pipeline.writers import JsonlWriter

    pipeline_exec = LocalPipelineExecutor(
        pipeline=[
            ParquetReader(
                src_dir,
                file_progress=True,
                glob_pattern="**/*.parquet",
            ),
            JsonlWriter(
                tgt_dir,
                output_filename=dataset + ".chunk.${rank}.jsonl",
                compression=None,
            ),
        ],
        tasks=ntasks,
        logging_dir=os.path.join(work_dir, "datatrove"),
    )
    pipeline_exec.run()


def setup_terashuf(work_dir):
    terashuf_dir = os.path.join(work_dir, "terashuf")
    terashuf_executable = os.path.join(terashuf_dir, "terashuf")

    if os.path.exists(terashuf_executable):
        print("terashuf executable already exists. Skipping setup.")
        return terashuf_dir

    print("Setting up terashuf...")
    run_command(f"git clone https://github.com/alexandres/terashuf {terashuf_dir}")
    run_command(f"make -C {terashuf_dir}")
    return terashuf_dir


def main(dataset):
    # Configuration
    repo_id = {
        "fineweb_edu": "HuggingFaceFW/fineweb-edu",
        "fineweb_edu_10bt": "HuggingFaceFW/fineweb-edu",
        "dclm_baseline_1.0": "mlfoundations/dclm-baseline-1.0",
    }[dataset]
    src_dir = f"data/{dataset}"
    out_dir = f"{src_dir}_shuffled"
    os.makedirs(out_dir, exist_ok=True)
    work_dir = src_dir  # Directory of this Python file
    prefix = f"{dataset}.chunk."
    orig_extension = {
        "fineweb_edu": ".jsonl",
        "fineweb_edu_10bt": ".jsonl",
        "dclm_baseline_1.0": ".jsonl.zst",
    }[dataset]
    cat_command = {
        "fineweb_edu": "cat",
        "fineweb_edu_10bt": "cat",
        "dclm_baseline_1.0": "zstdcat",
    }[dataset]
    allow_patterns = {
        "fineweb_edu": None,
        "fineweb_edu_10bt": "sample/10BT/*",
        "dclm_baseline_1.0": "*.jsonl.zst",
    }[dataset]
    suffix = ".jsonl"
    nchunks = 32
    k_validation = 10000  # Number of lines to take from each chunk for validation

    # Setup terashuf
    terashuf_dir = setup_terashuf(work_dir)

    # Download dataset
    download_dataset(repo_id, src_dir, allow_patterns)

    if "fineweb" in dataset:
        parquet_to_jsonl(dataset, work_dir, src_dir, src_dir)

    # Set up environment variables
    os.environ["MEMORY"] = "1000.0"
    os.environ["SEED"] = "42"

    # Run the original shuffling and splitting command
    terashuf_executable = os.path.join(terashuf_dir, "terashuf")
    run_command(
        f"ulimit -n 100000 && "
        f"find {src_dir} -type f -name '*{orig_extension}' -print0 | xargs -0 {cat_command} | {terashuf_executable} | "
        f"split -n r/{nchunks} -d --suffix-length 2 --additional-suffix {suffix} - {out_dir}/{prefix}"
    )

    # Create validation set and remove lines from chunks
    validation_file = f"{out_dir}/{dataset}.val{suffix}"
    for i in range(nchunks):
        chunk_file = f"{out_dir}/{prefix}{i:02d}{suffix}"
        run_command(f"head -n {k_validation} {chunk_file} >> {validation_file}")
        run_command(f"sed -i '1,{k_validation}d' {chunk_file}")

    print("All tasks completed successfully!")


if __name__ == "__main__":
    main(sys.argv[1])
