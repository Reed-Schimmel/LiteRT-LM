# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


include("${LITERTLM_MODULES_DIR}/utils.cmake")

set(ROOT_LIST "${LITERT_SRC_DIR}/CMakeLists.txt")
set(LITERTLM_LITERT_SHIM_PATH "${LITERT_PACKAGE_DIR}/litert_shims.cmake")


if(EXISTS "${ROOT_LIST}")
    message(STATUS "[LITERTLM PATCHER] Injecting shim into LiteRT root...")

    file(READ "${ROOT_LIST}" ROOT_CONTENT)
    string(REPLACE "project(LiteRT VERSION 1.4.0 LANGUAGES CXX C)"
           "project(LiteRT VERSION 1.4.0 LANGUAGES CXX C)\ninclude(${LITERTLM_LITERT_SHIM_PATH})"
           ROOT_CONTENT "${ROOT_CONTENT}")

    file(WRITE "${ROOT_LIST}" "${ROOT_CONTENT}")
else()
    message(STATUS "[LITERTLM] Root manifest already shimmed. Skipping injection.")
endif()


# --- 3. THE SUBMODULE GUILLOTINE ---
# Prevent LiteRT from entering these directories and triggering downloads/errors.

set(GUILLOTINE_PATHS
    "${LITERT_SRC_DIR}/third_party/tensorflow/CMakeLists.txt"
    "${LITERT_SRC_DIR}/tflite/CMakeLists.txt"
    "${LITERT_SRC_DIR}/tflite/tools/cmake/CMakeLists.txt"
)

foreach(TARGET_HEAD ${GUILLOTINE_PATHS})
    # Ensure directory exists so we can plant the dummy file
    get_filename_component(TARGET_DIR "${TARGET_HEAD}" DIRECTORY)
    if(NOT EXISTS "${TARGET_DIR}")
        file(MAKE_DIRECTORY "${TARGET_DIR}")
    endif()

    message(STATUS "[LITERTLM] Decapitating legacy build path: ${TARGET_HEAD}")
    file(WRITE "${TARGET_HEAD}" "message(STATUS \"[LITERTLM] Path Guillotined: ${TARGET_HEAD}\")\n")
endforeach()


# --- 4. TRANSITIVE TARGET REDIRECTION (Recursive Sweep) ---
# We scan for hardcoded paths and FetchContent calls that aliases can't fix.
file(GLOB_RECURSE ALL_CMAKELISTS "${LITERT_SRC_DIR}/*.cmake" "${LITERT_SRC_DIR}/**/CMakeLists.txt")

foreach(C_FILE ${ALL_CMAKELISTS})
    if("${C_FILE}" STREQUAL "${ROOT_LIST}")
        continue()
    endif()
    patch_file_content("${C_FILE}" "absl::[a-zA-Z0-9_]+" "LiteRTLM::absl::shim" TRUE)

    # A. Neutralize Hardcoded Flatbuffer Paths (Crucial)
    # TFLite/LiteRT loves to look for ".../flatbuffers-build/libflatbuffers.a" directly.
    patch_file_content("${C_FILE}" "[^\" ]*/_deps/flatbuffers-build/libflatbuffers.a" "LiteRTLM::flatbuffers::flatbuffers" TRUE)
    patch_file_content("${C_FILE}" "flatbuffers-build/libflatbuffers.a" "LiteRTLM::flatbuffers::flatbuffers" FALSE)
    patch_file_content("${C_FILE}" "TFLITE_FLATBUFFERS_LIB" "LiteRTLM::flatbuffers::flatbuffers" FALSE)

    # B. Kill Internal Discovery Logic
    # We want it to use the "flatc" target we defined in the Shim, not look for one.
    patch_file_content("${C_FILE}" "find_program\\(FLATC_EXECUTABLE[^\\)]+\\)" "# [LITERTLM] Suppressed: Using Global Shim" TRUE)
    # Ensure variables point to the target name "flatc", not a file path
    patch_file_content("${C_FILE}" "set\\(FLATC_EXECUTABLE \\$<TARGET_FILE:flatc>\\)" "set(FLATC_EXECUTABLE flatc)" TRUE)

    # C. Neutralize FetchContent (The "Anti-Download" Shield)
    patch_file_content("${C_FILE}" "FetchContent_Declare\\([^\\)]+\\)" "# [LITERTLM] Suppressed: External fetch prohibited" TRUE)
    patch_file_content("${C_FILE}" "FetchContent_MakeAvailable\\([^\\)]+\\)" "# [LITERTLM] Suppressed: Using Global Manifest" TRUE)
endforeach()


# --- 5. SOURCE-LEVEL REMEDIATION ---
# Correcting C++ API signature drifts.
patch_file_content("${LITERT_SRC_DIR}/runtime/compiled_model.cc"
    " return litert_cpu_buffer_requirements"
    "return litert::Expected<const LiteRtTensorBufferRequirementsT*>(litert_cpu_buffer_requirements)" FALSE)

# Fix directory structure mismatch
patch_file_content("${ROOT_LIST}" "add_subdirectory(compiler_plugin)" "add_subdirectory(compiler)" FALSE)


# --- 6. VENDOR SUBSYSTEM DECOUPLING ---
set(V_LIST "${LITERT_SRC_DIR}/vendors/CMakeLists.txt")
if(EXISTS "${V_LIST}")
    file(READ "${V_LIST}" V_CONTENT)

    # 1. Find the start of the MediaTek block
    string(FIND "${V_CONTENT}" "if(VENDOR STREQUAL \"MediaTek\")" START_POS)

    if(NOT START_POS EQUAL -1)
        # 2. Find the FIRST 'endif()' that occurs AFTER the start pos
        # We look for the anchor first to make sure we're in the right place
        string(FIND "${V_CONTENT}" "add_custom_target(mediatek_schema_gen" ANCHOR_POS)

        if(NOT ANCHOR_POS EQUAL -1)
            message(STATUS "[LITERTLM] Decoupling Vendor Dependencies (Surgical Slice)...")

            # Find the endif() relative to the anchor
            string(SUBSTRING "${V_CONTENT}" ${ANCHOR_POS} -1 POST_ANCHOR)
            string(FIND "${POST_ANCHOR}" "endif()" ENDIF_REL_POS)

            # Calculate total end position (Anchor + Relative End + 'endif()' length)
            math(EXPR END_POS "${ANCHOR_POS} + ${ENDIF_REL_POS} + 7")

            # 3. Slice it out
            string(SUBSTRING "${V_CONTENT}" 0 ${START_POS} PRE_BLOCK)
            string(SUBSTRING "${V_CONTENT}" ${END_POS} -1 POST_BLOCK)

            # 4. Stitch in the include
            set(INJECTION "\n# [LITERTLM] MediaTek Logic Virtualized\ninclude(\"${VENDOR_SHIM_PATH}\")\n")
            file(WRITE "${V_LIST}" "${PRE_BLOCK}${INJECTION}${POST_BLOCK}")
        endif()
    endif()
endif()


# --- 7. MANDATORY BUILD CONFIGURATION ---
# Enforce deterministic GPU/NPU flags via a generated header.
message(STATUS "[LITERTLM] Enforcing deterministic build_config.h...")
set(LITERT_GEN_DIR "${LITERT_SRC_DIR}/build_common")

if(NOT EXISTS "${LITERT_GEN_DIR}")
    file(MAKE_DIRECTORY "${LITERT_GEN_DIR}")
endif()

if(NOT DEFINED LITERT_BUILD_CONFIG_DISABLE_GPU_VAL)
    set(LITERT_BUILD_CONFIG_DISABLE_GPU_VAL 1)
endif()
if(NOT DEFINED LITERT_BUILD_CONFIG_DISABLE_NPU_VAL)
    set(LITERT_BUILD_CONFIG_DISABLE_NPU_VAL 1)
endif()

set(BUILD_CONFIG_CONTENT "/* Generated by LiteRTLM Patcher - Deterministic Configuration */
#ifndef LITE_RT_BUILD_COMMON_BUILD_CONFIG_H_
#define LITE_RT_BUILD_COMMON_BUILD_CONFIG_H_

#define LITERT_BUILD_CONFIG_DISABLE_GPU ${LITERT_BUILD_CONFIG_DISABLE_GPU_VAL}
#define LITERT_BUILD_CONFIG_DISABLE_NPU ${LITERT_BUILD_CONFIG_DISABLE_NPU_VAL}

#endif  /* LITE_RT_BUILD_COMMON_BUILD_CONFIG_H_ */\n")

file(WRITE "${LITERT_GEN_DIR}/build_config.h" "${BUILD_CONFIG_CONTENT}")


set(LAYOUT_HDR "${TFLITE_SRC_DIR}/../litert/cc/litert_layout.h")

if(EXISTS "${LAYOUT_HDR}")
    message(STATUS "[LITERTLM PATCHER] Neutralizing constexpr violation in litert_layout.h")

    file(READ "${LAYOUT_HDR}" CONTENT)
    string(REPLACE "constexpr LiteRTLayout" "inline LiteRTLayout" MODIFIED_CONTENT "${CONTENT}")

    if(NOT "${CONTENT}" STREQUAL "${MODIFIED_CONTENT}")
        file(WRITE "${LAYOUT_HDR}" "${MODIFIED_CONTENT}")
        message(STATUS "[LITERTLM PATCHER] Successfully converted constexpr constructor to inline.")
    else()
        message(WARNING "[LITERTLM PATCHER] Patch target not found in litert_layout.h. Check version compatibility.")
    endif()
endif()

message(STATUS "[LITERTLM] Surgical Patching Phase Complete.")