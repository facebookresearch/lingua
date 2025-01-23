load("@aspect_rules_py//py:defs.bzl", "py_binary", "py_library")

py_binary(
    name = "eval",
    srcs = ["eval.py"],
    imports = ["../.."],
    visibility = [
        "//:__subpackages__",
    ],
    deps = [
        ":generate",
        ":transformer",
        "//lingua:args",
        "//lingua:checkpoint",
        "//lingua:data",
        "//lingua:distributed",
        "@pip//lm_eval",
        "@pip//omegaconf",
        "@pip//torch",
    ],
)

py_binary(
    name = "generate",
    srcs = ["generate.py"],
    imports = ["../.."],
    visibility = [
        "//:__subpackages__",
    ],
    deps = [
        ":transformer",
        "//lingua:args",
        "//lingua:checkpoint",
        "//lingua:tokenizer",
        "//lingua:transformer",
        "@pip//omegaconf",
        "@pip//torch",
        "@pip//tqdm",
        "@pip//xformers",
    ],
)

py_binary(
    name = "train",
    srcs = ["train.py"],
    imports = ["../.."],
    visibility = [
        "//:__subpackages__",
    ],
    deps = [
        ":eval",
        ":transformer",
        "//lingua:args",
        "//lingua:checkpoint",
        "//lingua:data",
        "//lingua:distributed",
        "//lingua:logger",
        "//lingua:metrics",
        "//lingua:optim",
        "//lingua:probe",
        "//lingua:profiling",
        "//lingua:stool",
        "//lingua:tokenizer",
        "@pip//numpy",
        "@pip//omegaconf",
        "@pip//torch",
        "@pip//wandb",
        "@pip//xformers",
    ],
)

py_library(
    name = "transformer",
    srcs = ["transformer.py"],
    imports = ["../.."],
    visibility = [
        "//:__subpackages__",
    ],
    deps = [
        "//lingua:transformer",
        "@pip//torch",
        "@pip//xformers",
    ],
)
