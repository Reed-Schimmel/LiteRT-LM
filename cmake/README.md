# LiteRT-LM: CMake Overview
The LiteRT-LM CMake build system provides a unified infrastructure
for building all required third-party dependencies, internal
libraries, and the primary litert_lm_main executable.

Additional executable targets (such as
litert_lm_advanced_main) are defined
but currently gated behind the `_unverified_targets`
flag. They remain in an unverified state until they can be
validated within the standalone CMake environment.

## Dependency Management & Project Structure
This project implements a Super-Build pattern to ensure One Definition Rule
(ODR) adherence. This approach is necessary to manage a converging
dependency tree where multiple components rely on different versions
of the same core libraries.

### Dependency Strategy
The build system leverages a hybrid approach to dependency management:

- __FetchContent__: Used for conventional third-party libraries
where integration is sufficient
- __ExternalProject__: Reserved for dependencies requiring heavy
modification. These are orchestrated to redirect include and
library paths to a unified "source of truth" within the build
directory. This prevents the symbol collisions that occur when
multiple dependencies introduce conflicting versions of the same
provider (e.g., Abseil or Protobuf).

### Package Infrastructure
Orchestration logic for external dependencies is modularized within
`cmake/packages/<name>`. To ensure a hermetic build environment and
strict ODR adherence, these modules implement a Source
Transformation and Target Mapping framework.

This framework does not simply wrap dependencies; it actively
transforms them by "de-nesting" internal third-party code and
normalizing source-level paths to align with the
LiteRT-LM unified build structure.

```bash
cmake/packages/sentencepiece
├── sentencepiece.cmake            # Primary orchestration module (ExternalProject_Add)
├── sentencepiece_patcher.cmake    # Source-level transformation (Path normalization and dependency de-nesting)
├── sentencepiece_root_shim.cmake  # Injected logic for the package-level configuration
├── sentencepiece_src_shim.cmake   # Injected logic for the source-level build definitions
├── sentencepiece_aggregate.cmake  # Logic to consolidate build artifacts into a unified interface
└── sentencepiece_target_map.cmake # Dictionary mapping internal project targets to local static archives
```

#### Transformation Highlights

- __Hermeticity__: By removing nested third_party directories
within dependencies, we force all components to resolve a single,
verified version of core libraries (e.g., Abseil, Protobuf).

- __Path Normalization__: Source files are patched in-place to
canonicalize include paths, ensuring compatibility with the
standalone CMake layout.

- __Target Redirection__: Using custom mapping logic, internal targets
are transparently redirected to local static archives, maintaining
consistency with the original project structure without requiring a
full monorepo environment.

### Project Layout
To maintain parity with the internal codebase and facilitate automated
maintenance, LiteRT-LM modeled its CMake target definitions to
mirror the source tree.

- __Source-Locality__: Target definitions for LiteRT-LM components generally reside in the CMakeLists.txt file located in the same directory as their respective source files.

- __Exceptions__: Shared resources (such as proto_lib) are consolidated into centralized configuration files to manage global visibility and reuse

## Build Guide
This project targets a modern high-performance C++ environment. Currently,
the build system is strictly verified for the GNU toolchain on Debian-based
Linux (e.g., Ubuntu 24.04).

#### Prerequisites
Ensure your local environment meets these minimum version requirements to
avoid compilation errors related to C++20 standards and build-time
orchestration.

- __Compiler__: gcc / g++ 13+ (Required for stable C++20 feature support).

- __Build Tools__: cmake (3.25+) and make.

- __Python__: 3.12+ (Required development scripts).

- __Java__: openjdk-17-jre-headless or newer (Required for ANTLR 4).

- __Rust__: The Rust toolchain is required for specific sub-components.

- __System Libraries__: zlib1g-dev, libssl-dev libcurl4-openssl-dev.

---

__1. Configuration__

Create a build directory to maintain a clean source tree. Note that while
the build system is currently hard-coded to C++20, future updates will
transition this to a configurable variable.

```bash
cmake -B cmake/build -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_STANDARD=20
```

__2. Executing the Build__

Parallel execution is highly recommended to manage the complex dependency
tree, but it must be balanced against available system resources.

**WARNING** High Memory Usage: Allocating excessive parallel jobs can cause
a SEGFAULT or an OOM-kill (Signal 9). To ensure a stable build, use a
conservative job count.

__Recommended Formula__: Available RAM / 8GB = Max `-j` value

```bash
# Example for 32GB RAM
cmake --build cmake/build -t litert_lm_main -j4
```

__3. Verification__

Verify the binary integrity and Package Infrastructure mapping via a CPU-based
inference test.
This confirms that all internal symbols and external shims
(Abseil, Protobuf, etc.) are correctly linked and functional.

###### Validation Scope & Environment Notes

- __Primary Target:__ The current verification suite focuses on the
CPU-reference implementation,
leveraging the high-compute density (48-core) of the development environment.

- __Future Target:__ Verification of hardware-accelerated backends (GPU/NPU) is
deferred to environments with dedicated hardware resource reservation.
Validation requires a physical target or an instance with direct hardware
passthrough to establish the necessary compute-level interface.

```bash
./litert_lm_main \
  --model_path=/path/to/gemma-3n-E2B-it-int4.litertlm \
  --backend=cpu \
  --input_prompt="What is the tallest building in the world?"
```
__Expected Output__: A successful build will initialize the XNNPACK delegate and
return the model response along with benchmark metrics:

- _Init Phases_: Executor and Tokenizer initialization times.

- _Prefill/Decode Speeds_: Performance stats (tokens/sec) indicating the backend is optimized.

_Example_

```
dev-sh@:LiteRT-LM$ cmake/build/litert_lm_main --model_path=$model_path/gemma-3n-E2B-it-int4.litertlm --backend=cpu --input_prompt="What is the tallest building in the world?"
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
input_prompt: What is the tallest building in the world?
The tallest building in the world is the **Burj Khalifa** in Dubai, United Arab Emirates.

It stands at a staggering **828 meters (2,717 feet)** tall.

It was completed in 2010 and continues to hold the record.

BenchmarkInfo:
  Init Phases (2):
    - Executor initialization: 844.54 ms
    - Tokenizer initialization: 66.70 ms
    Total init time: 911.25 ms
--------------------------------------------------
  Time to first token: 2.40 s
--------------------------------------------------
  Prefill Turns (Total 1 turns):
    Prefill Turn 1: Processed 18 tokens in 2.311920273s duration.
      Prefill Speed: 7.79 tokens/sec.
--------------------------------------------------
  Decode Turns (Total 1 turns):
    Decode Turn 1: Processed 62 tokens in 5.53092314s duration.
      Decode Speed: 11.21 tokens/sec.
--------------------------------------------------
--------------------------------------------------
```

<br>

---
This project is licensed under the
[Apache 2.0 License.](https://github.com/google-ai-edge/LiteRT-LM/blob/main/LICENSE)