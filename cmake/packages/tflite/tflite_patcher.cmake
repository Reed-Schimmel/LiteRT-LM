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


message(STATUS "[LITERTLM] Starting surgical orchestration...")

set(CONFIG_GEN_H "${TENSORFLOW_SOURCE_DIR}/tensorflow/lite/acceleration/configuration/configuration_generated.h")
if(EXISTS "${CONFIG_GEN_H}")
    file(READ "${CONFIG_GEN_H}" CONTENT)
    string(REGEX REPLACE "FLATBUFFERS_VERSION_MAJOR == [0-9]+" "FLATBUFFERS_VERSION_MAJOR >= 25" CONTENT "${CONTENT}")
    file(WRITE "${CONFIG_GEN_H}" "${CONTENT}")
endif()

if(EXISTS "${LITERTLM_PROJECT_ROOT}/cmake/patches/litert_converter.zip")
    message(STATUS "[LITERTLM PATCHER] Extracting litert_converter.zip...")
    file(ARCHIVE_EXTRACT INPUT "${LITERTLM_PROJECT_ROOT}/cmake/patches/litert_converter.zip" DESTINATION "${TFLITE_SRC_DIR}")
endif()

set(TFLITE_CMAKELISTS "${TENSORFLOW_SOURCE_DIR}/tensorflow/lite/CMakeLists.txt")
if(EXISTS "${TFLITE_CMAKELISTS}")
    file(READ "${TFLITE_CMAKELISTS}" CONTENT)
    # Replace find_program(FLATC_BIN...) with a hardcoded set()
    string(REGEX REPLACE "find_program\\(FLATC_BIN flatc HINTS \\\${FLATC_PATHS}\\)"
           "set(FLATC_BIN \"${FLATC_EXECUTABLE}\" CACHE FILEPATH \"Forced by LiteRT-LM\")" CONTENT "${CONTENT}")
    file(WRITE "${TFLITE_CMAKELISTS}" "${CONTENT}")
endif()

set(SCHEMA_GEN_H "${TENSORFLOW_SOURCE_DIR}/tensorflow/compiler/mlir/lite/schema/schema_generated.h")
if(EXISTS "${SCHEMA_GEN_H}")
    file(READ "${SCHEMA_GEN_H}" CONTENT)
    string(REPLACE "FLATBUFFERS_VERSION_MAJOR == 24" "FLATBUFFERS_VERSION_MAJOR >= 24" CONTENT "${CONTENT}")
    string(REPLACE "FLATBUFFERS_VERSION_MINOR == 3" "FLATBUFFERS_VERSION_MINOR >= 0" CONTENT "${CONTENT}")
    string(REPLACE "FLATBUFFERS_VERSION_REVISION == 25" "FLATBUFFERS_VERSION_REVISION >= 0" CONTENT "${CONTENT}")
    file(WRITE "${SCHEMA_GEN_H}" "${CONTENT}")
endif()

set(_downloader_modules
    "tensorflow/lite/tools/cmake/modules/abseil-cpp.cmake"
    "tensorflow/lite/tools/cmake/modules/protobuf.cmake"
    "tensorflow/lite/tools/cmake/modules/flatbuffers.cmake"
)

foreach(_mod IN LISTS _downloader_modules)
    set(_mod_path "${TFLITE_SRC_DIR}/${_mod}")
    if(EXISTS "${_mod_path}")
        message(STATUS "[LITERTLM PATCHER] Neutralizing ${_mod}")
        file(READ "${_mod_path}" CONTENT)
        file(WRITE "${_mod_path}" "return()\n${CONTENT}")
    endif()
endforeach()

file(GLOB_RECURSE _tflite_cmakelists "${TFLITE_SRC_DIR}/CMakeLists.txt")

foreach(_list IN LISTS _tflite_cmakelists)
    file(READ "${_list}" CONTENT)
    string(REGEX REPLACE "[ \t\n\r]+absl::([a-zA-Z0-9_]+)" " LiteRTLM::absl::absl" CONTENT "${CONTENT}")
    string(REGEX REPLACE "[ \t\n\r]+protobuf::([a-zA-Z0-9_-]+)" " LiteRTLM::protobuf::libprotobuf" CONTENT "${CONTENT}")
    string(REGEX REPLACE "[ \t\n\r]+flatbuffers::([a-zA-Z0-9_-]+)" " LiteRTLM::flatbuffers::flatbuffers" CONTENT "${CONTENT}")
    file(WRITE "${_list}" "${CONTENT}")
endforeach()

set(V_FILES
    "tensorflow/lite/acceleration/configuration/configuration_generated.h"
    "tensorflow/compiler/mlir/lite/schema/schema_generated.h"
)

foreach(V_FILE ${V_FILES})
    set(FULL_PATH "${TENSORFLOW_SOURCE_DIR}/${V_FILE}")
    if(EXISTS "${FULL_PATH}")
        file(READ "${FULL_PATH}" CONTENT)
        string(REGEX REPLACE "FLATBUFFERS_VERSION_MAJOR == [0-9]+" "FLATBUFFERS_VERSION_MAJOR >= 24" CONTENT "${CONTENT}")
        string(REGEX REPLACE "FLATBUFFERS_VERSION_MINOR == [0-9]+" "FLATBUFFERS_VERSION_MINOR >= 0" CONTENT "${CONTENT}")
        file(WRITE "${FULL_PATH}" "${CONTENT}")
    endif()
endforeach()

set(MODULES "abseil-cpp.cmake" "protobuf.cmake" "flatbuffers.cmake")
foreach(MOD ${MODULES})
    set(MOD_PATH "${TENSORFLOW_SOURCE_DIR}/tensorflow/lite/tools/cmake/modules/${MOD}")
    if(EXISTS "${MOD_PATH}")
        file(WRITE "${MOD_PATH}" "return()\n")
    endif()
endforeach()

file(GLOB_RECURSE ALL_CMAKELISTS "${TFLITE_SRC_DIR}/CMakeLists.txt")

foreach(C_FILE ${ALL_CMAKELISTS})
    file(READ "${C_FILE}" CONTENT)
    string(REGEX REPLACE "[ \t]absl::[a-zA-Z0-9_]+" " LiteRTLM::absl::absl" CONTENT "${CONTENT}")
    string(REGEX REPLACE "[ \t]protobuf::[a-zA-Z0-9_-]+" " LiteRTLM::protobuf::libprotobuf" CONTENT "${CONTENT}")
    string(REGEX REPLACE "[ \t]flatbuffers::[a-zA-Z0-9_-]+" " LiteRTLM::flatbuffers::flatbuffers" CONTENT "${CONTENT}")
    file(WRITE "${C_FILE}" "${CONTENT}")
endforeach()

set(XNNPACK_MOD_FILE "${TFLITE_SRC_DIR}/tools/cmake/modules/xnnpack/CMakeLists.txt")

if(EXISTS "${XNNPACK_MOD_FILE}")
    message(STATUS "[LITERTLM] Automating XNNPACK path neutralization...")
    file(APPEND "${XNNPACK_MOD_FILE}" \
    "# --- LiteRT-LM Automated Fix ---
    execute_process(
        COMMAND find \"\${xnnpack_SOURCE_DIR}\" -type f -exec sed -i \"s|flatbuffers-flatc/bin/flatc|\${FLATBUFFERS_FLATC_EXECUTABLE}|g\" {} +
    )"
    )
endif()

set(XNN_DELEGATE_CMAKELISTS "${TFLITE_SRC_DIR}/CMakeLists.txt")

if(EXISTS "${XNN_DELEGATE_CMAKELISTS}")
    message(STATUS "[LITERTLM] Hard-patching XNNPACK delegate custom command...")
    execute_process(
        COMMAND sed -i "s|\"\${FLATBUFFERS_FLATC_EXECUTABLE}\"|\"${FLATC_EXECUTABLE}\"|g" "${XNN_DELEGATE_CMAKELISTS}"
        COMMAND sed -i "s|\"\${FLATC_TARGET}\"|\"${FLATC_EXECUTABLE}\"|g" "${XNN_DELEGATE_CMAKELISTS}"
    )
endif()

set(ROOT_LIST "${TFLITE_SRC_DIR}/CMakeLists.txt")
file(READ "${ROOT_LIST}" CONTENT)

string(REPLACE "project(tensorflow-lite C CXX)"
    "project(tensorflow-lite C CXX)\ninclude(${LITERTLM_PACKAGES_DIR}/tflite/tflite_shims.cmake)"
    CONTENT "${CONTENT}")
file(WRITE "${ROOT_LIST}" "${CONTENT}")


set(XNN_CMAKELISTS "${TFLITE_SRC_DIR}/CMakeLists.txt")

message(STATUS "[LITERTLM] Performing manual XNNPACK schema generation...")

execute_process(
    COMMAND "${FLATC_EXECUTABLE}" -c
            -o "${TFLITE_SRC_DIR}/"
            --gen-mutable --gen-object-api
            "${TFLITE_SRC_DIR}/delegates/xnnpack/weight_cache_schema.fbs"
    RESULT_VARIABLE manual_gen_res
)

if(NOT manual_gen_res EQUAL 0)
    message(FATAL_ERROR "LITERTLM: Manual flatc generation failed! Path: ${FLATC_EXECUTABLE}")
endif()

execute_process(
    COMMAND sed -i "/add_custom_command(/,/)/d" "${XNN_CMAKELISTS}"
)

file(MAKE_DIRECTORY "${TFLITE_SRC_DIR}/delegates/xnnpack")
file(COPY "${TFLITE_SRC_DIR}/weight_cache_schema_generated.h"
     DESTINATION "${TFLITE_BUILD_DIR}/tensorflow/lite/delegates/xnnpack")


set(PROTO_RECORDS
    "tensorflow/lite/profiling/proto/CMakeLists.txt:profiling_info.proto"
    "tensorflow/lite/tools/benchmark/proto/CMakeLists.txt:benchmark_result.proto"
)

foreach(RECORD ${PROTO_RECORDS})
    string(REPLACE ":" ";" FIELDS ${RECORD})
    list(GET FIELDS 0 PLIST)
    list(GET FIELDS 1 PFILE)
    set(TARGET_LIST "${TENSORFLOW_SOURCE_DIR}/${PLIST}")

    if(EXISTS "${TARGET_LIST}")
        message(STATUS "[LITERTLM] Applying surgical fix to ${PLIST}...")

        get_filename_component(PDIR "${PLIST}" DIRECTORY)

        execute_process(COMMAND sed -i "s|--proto_path=[^ ]*|--proto_path=${TENSORFLOW_SOURCE_DIR}|g" "${TARGET_LIST}")

        execute_process(COMMAND sed -i "s| [^ ]*${PFILE}| ${TENSORFLOW_SOURCE_DIR}/${PDIR}/${PFILE}|g" "${TARGET_LIST}")

        execute_process(COMMAND sed -i "s|tflite/|tensorflow/lite/|g" "${TARGET_LIST}")
    endif()
endforeach()