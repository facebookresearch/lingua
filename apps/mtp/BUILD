load("@aspect_rules_py//py:defs.bzl", "py_binary", "py_library")

py_binary(
    name = "eval",
    srcs = ["eval.py"],
    imports = ["../.."],
    visibility = ["//lingua_repo:__subpackages__"],
    deps = [
        ":transformer",
        "//lingua_repo/apps/main:eval",
        "//lingua_repo/apps/main:generate",
        "//lingua_repo/lingua:args",
        "//lingua_repo/lingua:checkpoint",
        "//lingua_repo/lingua:distributed",
        "@pip//lm_eval",
        "@pip//omegaconf",
        "@pip//torch",
    ],
)

py_binary(
    name = "train",
    srcs = ["train.py"],
    imports = ["../.."],
    visibility = ["//lingua_repo:__subpackages__"],
    deps = [
        ":eval",
        ":transformer",
        "//lingua_repo/lingua:args",
        "//lingua_repo/lingua:checkpoint",
        "//lingua_repo/lingua:data",
        "//lingua_repo/lingua:distributed",
        "//lingua_repo/lingua:logger",
        "//lingua_repo/lingua:metrics",
        "//lingua_repo/lingua:optim",
        "//lingua_repo/lingua:probe",
        "//lingua_repo/lingua:profiling",
        "//lingua_repo/lingua:stool",
        "//lingua_repo/lingua:tokenizer",
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
    visibility = ["//lingua_repo:__subpackages__"],
    deps = [
        "//lingua_repo/lingua:transformer",
        "@pip//torch",
        "@pip//xformers",
    ],
)
