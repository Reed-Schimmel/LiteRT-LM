# OpenCode / Agent Instructions for LiteRT-LM

LiteRT-LM is a cross-platform (C++, Python, Kotlin, Rust) monorepo for deploying Large Language Models on edge devices.

## Build and Toolchain
* **Bazel** is the primary build system (v7.6.1 via Bazelisk). Avoid standard `cmake` or `make` commands.
* **Prebuilts:** Run `git lfs pull` to fetch required prebuilt libraries in `prebuilt/` before running GPU builds or tests.
* **Primary App Entrypoint:** The core CLI binary used for local demo/execution is `litert_lm_main.cc`, located in `runtime/engine/`.

## Common Commands
* **Build Core CLI (Linux/Mac CPU):** `bazel build //runtime/engine:litert_lm_main`
* **Build GPU (Dynamic Linking):** Append `--define=litert_link_capi_so=true --define=resolve_symbols_in_exec=false` to the bazel build command.
* **Run Bazel C++ Unit Tests:** `bazel test --test_output=errors //...`
* **Run E2E Python Tests:** Requires a `.litertlm` model file and `pytest==8.3.4`. 
  `pytest tools/test/ --model-path=<path_to_.litertlm> --build-system=bazel`
* **Android Cross-compile:** Requires `ANDROID_NDK_HOME` set to NDK r28b+. Run: `bazel build --config=android_arm64 //runtime/engine:litert_lm_main`

## Architecture Boundaries
* `runtime/`: Core C++ framework, executor, and engine components.
* `c/`: C API for the framework.
* `python/`: Python SDK bindings and the `litert_lm` tool.
* `kotlin/`: JVM and Android SDKs.
* `rust/`: Hugging Face Tokenizers dependencies integrated via `cxxbridge`.

## Quirks & Notes
* If modifying GPU backend logic, note that you will need shared libraries from `prebuilt/<os>/` copied to the same directory as the executable to test properly.
* The repo makes heavy use of `.bzl` rules (`PATCH.*`, `WORKSPACE`, `rust_cxx_bridge.bzl`) to pull from upstream Google repositories like TensorFlow. When modifying build configs, ensure you check for patched behavior in the root directory.