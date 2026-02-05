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
include("${ABSL_PACKAGE_DIR}/absl_aggregate.cmake")

add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)

generate_absl_aggregate()

set(protobuf_ABSL_PROVIDER "package" CACHE INTERNAL "" FORCE)
set(protobuf_ABSL_USED_TARGETS "LiteRTLM::absl::absl" CACHE INTERNAL "" FORCE)
set(protobuf_ABSL_USED_TEST_TARGETS "LiteRTLM::absl::absl" CACHE INTERNAL "" FORCE)


include_directories(${ABSL_INCLUDE_DIR})
link_libraries(LiteRTLM::absl::shim)


set(CMAKE_CXX_STANDARD_LIBRARIES
    "${CMAKE_CXX_STANDARD_LIBRARIES} -Wl,--start-group ${_ABSL_PAYLOAD} -lz -lrt -lpthread -ldl -Wl,--end-group"
    CACHE STRING "Forced Abseil aggregate for Protobuf internal linking" FORCE
)

add_definitions(-DABSL_LTS_GROUP_EXPORT)
add_definitions(-DABSL_20250814_LTS)