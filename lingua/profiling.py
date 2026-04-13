# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import contextlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import torch.distributed
import wandb
import xformers.profiler
from torch.profiler.profiler import profile
from xformers.profiler import (
    MemSnapshotsProfiler,
    PyTorchProfiler,
)

from lingua.distributed import get_is_master


@dataclass
class ProfilerArgs:
    run: bool = False
    trace_folder: str = "profiling"
    mem_warmup: int = 100
    mem_steps: int = 2
    profile_warmup: int = 102
    profile_steps: int = 2


logger = logging.getLogger()


def perfetto_to_html(json_file, html_file):
    import gzip
    import string

    import viztracer

    root = os.path.dirname(viztracer.__file__)
    sub = {}
    json_file = gzip.open(json_file) if ".gz" in str(json_file) else open(json_file)
    with open(
        os.path.join(root, "html/trace_viewer_embedder.html"), encoding="utf-8"
    ) as f:
        tmpl = f.read()
    with open(os.path.join(root, "html/trace_viewer_full.html"), encoding="utf-8") as f:
        sub["trace_viewer_full"] = f.read()
    with json_file as j:
        content = j.read()
        if isinstance(content, bytes):
            content = content.decode("utf-8")
        sub["json_data"] = content.replace("</script>", "<\\/script>")  # type: ignore
    with open(html_file, "w+", encoding="utf-8") as output_file:
        output_file.write(string.Template(tmpl).substitute(sub))


class PyTorchProfilerWandb(PyTorchProfiler):
    def __init__(self, main_profiler) -> None:
        self.main_profiler = main_profiler
        self.num_steps = 0
        self.pytorch_profiler = torch.profiler.profile(
            on_trace_ready=self._on_trace,
            profile_memory=True,
            record_shapes=True,
            # With stack gives huge profile traces
            # and bugs out because of some non ascii
            # character somewhere in pytorch
            with_stack=False,
            with_flops=True,
            activities=self.ACTIVITIES,
        )

    def _analyze_trace(self, prof: profile):
        logger.info("Begin analyze trace")
        super()._analyze_trace(prof)
        logger.info("End analyze trace")

    def _on_trace(self, prof: torch.profiler.profiler.profile) -> None:
        super()._on_trace(prof)
        if get_is_master() and wandb.run is not None:
            filename = list(
                Path(self.main_profiler.output_dir).glob(
                    "profile_CPU_CUDA*/*.pt.trace.json*"
                )
            )[0]
            html_path = str(filename).replace(".json", ".html")
            perfetto_to_html(filename, html_path)
            wandb.log({"profile_trace": wandb.Html(html_path)})


class MemSnapshotsProfilerWandb(MemSnapshotsProfiler):
    def __exit__(self, exc_type, exc_val, exc_tb):
        super().__exit__(exc_type, exc_val, exc_tb)
        if get_is_master() and wandb.run is not None:
            filename = list(
                Path(self.main_profiler.output_dir).glob("memory_trace_plot/*.html")
            )[0]
            wandb.log({"memory_trace": wandb.Html(open(filename), inject=False)})


@contextlib.contextmanager
def maybe_run_profiler(dump_dir, module, config: ProfilerArgs):
    # get user defined profiler settings

    if config.run:
        trace_dir = os.path.join(dump_dir, config.trace_folder)

        logger.info(f"Profiling active.  Traces will be saved at {trace_dir}")

        if get_is_master() and not os.path.exists(trace_dir):
            os.makedirs(trace_dir)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        with xformers.profiler.profile(
            output_dir=trace_dir,
            module=module,
            schedule=[
                (
                    MemSnapshotsProfilerWandb,
                    config.mem_warmup,
                    config.mem_warmup + config.mem_steps,
                ),
                (
                    PyTorchProfilerWandb,
                    config.profile_warmup,
                    config.profile_warmup + config.profile_steps,
                ),
            ],
        ) as profiler:
            yield profiler

    else:
        torch_profiler = contextlib.nullcontext()
        yield None
