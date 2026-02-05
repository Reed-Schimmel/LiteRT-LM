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


set(FIXED_FLATC "${FLATC_EXECUTABLE}" CACHE INTERNAL "Forced" FORCE)
set(FLATC_TARGET                 "${FIXED_FLATC}" CACHE INTERNAL "Forced" FORCE)
set(FLATC_BIN                    "${FIXED_FLATC}" CACHE INTERNAL "Forced" FORCE)
set(FLATBUFFERS_FLATC_EXECUTABLE "${FIXED_FLATC}" CACHE INTERNAL "Forced" FORCE)
set(flatbuffers_FLATC_EXECUTABLE "${FIXED_FLATC}" CACHE INTERNAL "Forced" FORCE)
set(TFLITE_HOST_TOOLS_DIR        "${FIXED_FLATC}" CACHE PATH     "Forced" FORCE)
set(FLATC_PATHS                  "${FIXED_FLATC}" CACHE STRING   "Forced" FORCE)
set(flatbuffers_FOUND            TRUE             CACHE INTERNAL "Forced" FORCE)
set(FlatBuffers_FOUND            TRUE             CACHE INTERNAL "Forced" FORCE)

if(NOT TARGET LiteRTLM::flatbuffers::flatbuffers)
    add_library(LiteRTLM::flatbuffers::flatbuffers INTERFACE IMPORTED GLOBAL)
    set_target_properties(LiteRTLM::flatbuffers::flatbuffers PROPERTIES
        INTERFACE_LINK_LIBRARIES "imp_flatbuffers"
        INTERFACE_INCLUDE_DIRECTORIES "${FLATBUFFERS_INCLUDE_DIR}"
    )
    if(NOT TARGET flatbuffers::flatbuffers)
        add_library(flatbuffers::flatbuffers ALIAS LiteRTLM::flatbuffers::flatbuffers)
    endif()
endif()

foreach(_target_name flatbuffers-flatc flatbuffers-flatc-NOTFOUND)
    if(NOT TARGET ${_target_name})
        add_executable(${_target_name} IMPORTED GLOBAL)
        set_target_properties(${_target_name} PROPERTIES
            IMPORTED_LOCATION "${FIXED_FLATC}"
        )
    endif()
endforeach()

set(RELATIVE_PROBLEM_PATH "flatbuffers-flatc/bin/flatc")

execute_process(
    COMMAND find "${TENSORFLOW_SOURCE_DIR}/tensorflow/lite" -name "*.cmake" -o -name "CMakeLists.txt"
    -exec sed -i "s|${RELATIVE_PROBLEM_PATH}|${FLATC_EXECUTABLE}|g" {} +
)

if(EXISTS "${TFLITE_BUILD_DIR}/_deps")
    execute_process(
        COMMAND find "${TFLITE_BUILD_DIR}/_deps" -name "*.cmake" -o -name "CMakeLists.txt"
        -exec sed -i "s|${RELATIVE_PROBLEM_PATH}|${FLATC_EXECUTABLE}|g" {} +
    )
endif()
