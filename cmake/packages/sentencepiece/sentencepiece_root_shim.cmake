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

message(STATUS "[LiteRTLM] Redirecting SentencePiece dependencies...")

set(SPM_USE_BUILTIN_PROTOBUF OFF CACHE BOOL "" FORCE)
set(SPM_PROTOBUF_PROVIDER "package" CACHE STRING "" FORCE)
set(SPM_ABSL_PROVIDER "package" CACHE STRING "" FORCE)
set(Protobuf_INCLUDE_DIRS ${PROTO_INCLUDE_DIR} CACHE PATH "" FORCE)
set(PROTOBUF_INCLUDE_DIR ${PROTO_INCLUDE_DIR} CACHE PATH "" FORCE)
set(ABSL_SRC_FILE_PATH "${ABSL_SRC_DIR}/absl" CACHE PATH "" FORCE)
set(ABSL_INLUDE_FILE_PATH "${ABSL_INCLUDE_DIR}" CACHE PATH "" FORCE)

include("${LITERTLM_MODULES_DIR}/utils.cmake")
include("${ABSL_PACKAGE_DIR}/absl_aggregate.cmake")
include("${PROTOBUF_PACKAGE_DIR}/protobuf_aggregate.cmake")

add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)

generate_absl_aggregate()
generate_protobuf_aggregate()

include_directories(${ABSL_INCLUDE_DIR} ${PROTO_INCLUDE_DIR})

set(CMAKE_CXX_STANDARD_LIBRARIES "${CMAKE_CXX_STANDARD_LIBRARIES} -Wl,--allow-multiple-definition -Wl,--start-group ${_PROTOBUF_PAYLOAD} ${_ABSL_PAYLOAD} -lz -lrt -lpthread -ldl -Wl,--end-group"
    CACHE STRING "" FORCE
)
