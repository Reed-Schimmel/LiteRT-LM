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


include_guard(GLOBAL)

include("${LITERTLM_MODULES_DIR}/utils.cmake")
include("${LITERTLM_PACKAGES_DIR}/packages.cmake")
include("${ABSL_PACKAGE_DIR}/absl_aggregate.cmake")
include("${PROTOBUF_PACKAGE_DIR}/protobuf_aggregate.cmake")
include("${FLATBUFFERS_PACKAGE_DIR}/flatbuffers_aggregate.cmake")
include("${TFLITE_PACKAGE_DIR}/tflite_aggregate.cmake")

add_link_options("-Wl,--allow-multiple-definition")

generate_absl_aggregate()
generate_protobuf_aggregate()
generate_flatbuffers_aggregate()
generate_flatc_aggregate()
generate_tflite_aggregate()

set(VENDOR_SHIM_PATH "${LITERT_PACKAGE_DIR}/shims/vendor_shim.cmake")

message(STATUS "[LITERTLM-SHIM] Initializing Dependency Virtualization...")

if(NOT TARGET nlohmann_json::nlohmann_json)
    add_library(nlohmann_json::nlohmann_json INTERFACE IMPORTED GLOBAL)
    set_target_properties(nlohmann_json::nlohmann_json PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${JSON_INCLUDE_DIR}/include"
    )
    message(STATUS "[LITERTLM-SHIM] Virtualized nlohmann_json::nlohmann_json")
endif()

if(NOT TARGET flatc)
    add_executable(flatc IMPORTED GLOBAL)
    set_target_properties(flatc PROPERTIES IMPORTED_LOCATION "${FLATC_EXECUTABLE}")
endif()
