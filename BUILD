load("@aspect_rules_py//py:defs.bzl", "py_binary")
load("@bazel_gazelle//:def.bzl", "gazelle")
load("@pip//:requirements.bzl", "all_whl_requirements")
load("@rules_python//python:defs.bzl", "py_library")
load("@rules_python//python:pip.bzl", "compile_pip_requirements")
load("@rules_python_gazelle_plugin//manifest:defs.bzl", "gazelle_python_manifest")
load("@rules_python_gazelle_plugin//modules_mapping:def.bzl", "modules_mapping")

compile_pip_requirements(
    name = "requirements",
    src = "requirements.in",
    requirements_txt = "requirements_lock.txt",
    requirements_windows = "requirements_windows.txt",
)

exports_files(["requirements_lock.txt"])

modules_mapping(
    name = "modules_map",
    wheels = all_whl_requirements,
)

# bazel run //:gazelle_python_manifest.update
gazelle_python_manifest(
    name = "gazelle_python_manifest",
    modules_mapping = ":modules_map",
    pip_repository_name = "pip",
    requirements = "requirements_lock.txt",
)

# bazel run //:gazelle
gazelle(
    name = "gazelle",
    gazelle = "@rules_python_gazelle_plugin//python:gazelle_binary",
)

# gazelle:exclude apps/mamba
# gazelle:exclude apps/plots
# gazelle:exclude apps/fastRNN

py_binary(
    name = "download_prepare_hf_data",
    srcs = ["setup/download_prepare_hf_data.py"],
    visibility = ["//:__subpackages__"],
    deps = [
        "@pip//datatrove",
        "@pip//huggingface_hub",
        "@pip//requests",
    ],
)

py_binary(
    name = "download_tokenizer",
    srcs = ["setup/download_tokenizer.py"],
    visibility = ["//:__subpackages__"],
    deps = [
        "@pip//huggingface_hub",
        "@pip//requests",
        "@pip//tiktoken",
    ],
)

py_library(
    name = "lingua",
    srcs = [
        "setup/download_prepare_hf_data.py",
        "setup/download_tokenizer.py",
    ],
    visibility = ["//:__subpackages__"],
    deps = [
        "@pip//datatrove",
        "@pip//huggingface_hub",
        "@pip//requests",
        "@pip//tiktoken",
    ],
)
