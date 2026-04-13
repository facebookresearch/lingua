# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Usage:
python probe_animation.py /path/to/xp [/path/to/other/xp ...]
Where `/path/to/xp` is a folder containing `probe/probe.0.jsonl`
"""

import argparse
import json
import math
from multiprocessing import Pool
from pathlib import Path

import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc

parser = argparse.ArgumentParser()
parser.add_argument("probe_folders", help="contains a `probe` folder", nargs="+")
args = parser.parse_args()

rc("animation", html="jshtml")
matplotlib.rcParams["animation.embed_limit"] = 2**128

DEFAULT_QUANTILES = [0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 0.999]
DATAS_PER_FILE = {}
datas = []
NUM_LAYERS = 0  # TODO: Deduce from files

for f in args.probe_folders:
    name = Path(f).name
    print("Loading ", name)
    file = None
    for probe_test in ["probe/probe.0.jsonl", "probe.json"]:
        if (Path(f) / probe_test).exists():
            file = Path(f) / probe_test
    assert file is not None, "Could not find probe json file"
    data = file.read_text()
    datas = []
    for line in data.splitlines():
        if line == "":
            continue
        datas.append(json.loads(line))
        datas[-1].setdefault("quantiles", DEFAULT_QUANTILES)

    DATAS_PER_FILE[name] = datas
    # Assumes layers have the form
    # `FSDP.module.blocks.{LAYER_NUM}...`

    NUM_LAYERS = max(
        NUM_LAYERS,
        1
        + max(
            int(k.split("FSDPTransformer.layers.", 1)[1].split(".")[0])
            for k in datas[0]["data"].keys()
            if k.startswith("FSDPTransformer.layers.") and k.endswith("::w")
        ),
    )
    for d in datas:
        d["meta"]["it"] = d["meta"]["global_step"]
    assert NUM_LAYERS > 0, "Couldn't deduce the model depth"


def get_mean_quantiles(df, names):
    means = []
    quantiles = [[] for _ in df["quantiles"]]
    for name in names:
        d = df["data"][name]
        for qi, qval in enumerate(d["quantiles"]):
            quantiles[qi].append(qval)
        means.append(d["mean"])
    return np.array(means), np.array(quantiles)


# m, q = get_mean_quantiles(datas[0], ["FSDP.module.blocks.22.mlp.fc2::in", "FSDP.module.blocks.23.mlp.fc2::in"])
# possible_keys = {k.split("::")[0] for k in datas[0]["data"].keys() if "blocks.0." in k}
# possible_suff = {k.split("::")[1] for k in datas[0]["data"].keys() if "blocks.0." in k}
# print("\n".join([str(x) for x in possible_keys]))
# print("\n".join([str(x) for x in possible_suff]))

COLORS = [f"tab:{x}" for x in ["blue", "orange", "green", "red"]]


class Plotter:
    def __init__(self, ax, run, layers, color, timesteps) -> None:
        datas = DATAS_PER_FILE[run]
        for i in timesteps:
            for layer in layers:
                if layer not in datas[i]["data"]:
                    raise ValueError(f"Run `{run}`: layer `{layer}` not found!")
        self.x = np.arange(0, len(layers), 1)
        (self.mean,) = ax.plot(self.x, self.x, color=color, label=run)
        self.fills = []
        self.animate_data = [get_mean_quantiles(datas[i], layers) for i in timesteps]
        self.iters = [datas[i]["meta"]["it"] for i in timesteps]
        self.minimum = min(
            [
                min(np.nanmin(quants[3]), np.nanmin(means))
                for means, quants in self.animate_data
            ]
        )
        self.maximum = max(
            [
                max(np.nanmax(quants[-4]), np.nanmax(means))
                for means, quants in self.animate_data
            ]
        )
        if not math.isfinite(self.minimum) or not math.isfinite(self.maximum):
            raise ValueError(
                f"Layer `{layers[0]}`: invalid min/max computed: {self.minimum}/{self.maximum}"
            )
        self.ax = ax
        self.color = color
        self.run_name = run

    def animate(self, i):
        for f in self.fills:
            f.remove()
        self.fills.clear()
        means, quants = self.animate_data[i]
        self.fills += [
            self.ax.fill_between(
                x=self.x, y1=quants[j], y2=quants[-1 - j], alpha=0.2, color=self.color
            )
            for j in [0, 1, 2, 3, 4, 5]
        ]
        self.mean.set_ydata(means)
        self.mean.set_label(f"{self.run_name} it={self.iters[i]}")
        return self.mean


def plot_depth_distr_time(layer_fmt, to_file=None, runs=None, subsample=8):
    plt.ioff()
    while isinstance(layer_fmt, str) or isinstance(layer_fmt[0], str):
        layer_fmt = [layer_fmt]

    if layer_fmt[0][0].format(0) not in datas[0]["data"].keys():
        return

    if runs is None:
        runs = list(DATAS_PER_FILE.keys())
    LAYERS = list(range(NUM_LAYERS))
    nrows = len(layer_fmt)
    ncols = len(layer_fmt[0])
    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        squeeze=False,
        sharex=True,
        figsize=[min(6 * ncols, 14), min(5 * nrows, 8)],
        layout="compressed",
    )
    timesteps = range(0, len(datas), subsample)
    plotters = {}
    for i in range(nrows):
        for j in range(ncols):
            print(layer_fmt[i][j])
            plotters[(i, j)] = [
                Plotter(
                    axs[i, j],
                    run,
                    [layer_fmt[i][j].format(layer) for layer in LAYERS],
                    color,
                    timesteps,
                )
                for run, color in zip(runs, COLORS)
            ]
            minimum = min(p.minimum for p in plotters[(i, j)])
            maximum = max(p.maximum for p in plotters[(i, j)])
            axs[i, j].set_ylim([minimum, maximum])

    def animate(t):
        out = []
        axs[0, 0].legend(loc="upper right")
        for k, v in plotters.items():
            i, j = k
            axs[i, j].set_title(layer_fmt[i][j])
            if i == nrows - 1:
                axs[i, j].set_xlabel("depth")
            for p in v:
                out.append(p.animate(t))
        return out

    ani = animation.FuncAnimation(
        fig, animate, frames=len(timesteps), interval=500, blit=True
    )
    if to_file is not None:
        print("Writing to", to_file)
        Path(to_file).write_text(ani.to_jshtml())


OUTPUT_FOLDER_NAME = "_AND_".join([Path(f).name for f in args.probe_folders])
RENDER_OUT_PATH = (Path("render_out") / OUTPUT_FOLDER_NAME).absolute()
RENDER_OUT_PATH.mkdir(parents=True, exist_ok=True)


def _render_attn():
    to_file = RENDER_OUT_PATH / "attention.html"
    plot_depth_distr_time(
        [
            [
                "FSDPTransformer.layers.{}.attention::attn_logits",
                "FSDPTransformer.layers.{}.attention::attn_entropy",
                "FSDPTransformer.layers.{}.attention.wo::in",
            ],
            [
                "FSDPTransformer.layers.{}.attention.wq::out",
                "FSDPTransformer.layers.{}.attention.wk::out",
                "FSDPTransformer.layers.{}.attention.wv::out",
            ],
        ],
        to_file=to_file,
        subsample=1,
    )


def _render_res():
    print("## RESIDUAL")
    to_file = RENDER_OUT_PATH / "residual.html"
    plot_depth_distr_time(
        [
            [
                # "FSDPTransformer.layers.{}::res_ffn",
                # "FSDPTransformer.layers.{}::res_attn",
                "FSDPTransformer.layers.{}.feed_forward.w3::out",
                "FSDPTransformer.layers.{}.attention.wo::out",
            ],
            [
                "FSDPTransformer.layers.{}::out",
                "FSDPTransformer.layers.{}::out.g",
            ],
        ],
        to_file=to_file,
        subsample=1,
    )


def _render_to_file(linear_layer: str):
    if linear_layer == "__res__":
        return _render_res()
    if linear_layer == "__attn__":
        return _render_attn()
    if f"{linear_layer.format(0)}::out" not in datas[0]["data"].keys():
        return
    print("## ", linear_layer)
    to_file = linear_layer.split("{}", 1)[-1].replace(".", "") + ".html"
    to_file = RENDER_OUT_PATH / to_file
    plot_depth_distr_time(
        [
            [f"{linear_layer}::{suffix}" for suffix in ["in", "w", "out"]],
            [f"{linear_layer}::{suffix}.g" for suffix in ["in", "w", "out"]],
        ],
        to_file=to_file,
        subsample=1,
    )


with Pool(10) as p:
    p.map(
        _render_to_file,
        [
            "__attn__",
            # DINO
            "FSDP.module.blocks.{}.mlp.fc1",
            "FSDP.module.blocks.{}.mlp.fc2",
            "FSDP.module.blocks.{}.mlp.qkv",
            "FSDP.module.blocks.{}.mlp.proj",
            # linguas
            "FSDPTransformer.layers.{}.attention.wq",
            "FSDPTransformer.layers.{}.attention.wk",
            "FSDPTransformer.layers.{}.attention.wv",
            "FSDPTransformer.layers.{}.attention.wo",
            "FSDPTransformer.layers.{}.feed_forward.w1",
            "FSDPTransformer.layers.{}.feed_forward.w2",
            "FSDPTransformer.layers.{}.feed_forward.w3",
            "__res__",
        ],
    )
