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


include(ExternalProject)

set(FLATBUFFERS_EXT_PREFIX ${EXTERNAL_PROJECT_BINARY_DIR}/flatbuffers CACHE INTERNAL "")
set(FLATBUFFERS_INSTALL_PREFIX ${FLATBUFFERS_EXT_PREFIX}/install CACHE INTERNAL "")
set(FLATBUFFERS_INCLUDE_DIR ${FLATBUFFERS_INSTALL_PREFIX}/include CACHE INTERNAL "")
set(FLATBUFFERS_SRC_DIR ${FLATBUFFERS_EXT_PREFIX}/src CACHE INTERNAL "")
set(FLATBUFFERS_BIN_DIR ${FLATBUFFERS_INSTALL_PREFIX}/bin CACHE INTERNAL "")
set(FLATBUFFERS_LIB_DIR ${FLATBUFFERS_INSTALL_PREFIX}/lib CACHE INTERNAL "")


set(FLATBUFFERS_DIR ${FLATBUFFERS_LIB_DIR}/cmake/flatbuffers CACHE INTERNAL "")
set(FLATBUFFERS_CMAKE_CONFIG_FILE ${FLATBUFFERS_LIB_DIR}/cmake/flatbuffers/flatbuffers-config.cmake CACHE INTERNAL "")
set(FLATC_EXECUTABLE ${FLATBUFFERS_BIN_DIR}/flatc CACHE INTERNAL "" FORCE)

setup_external_install_structure("${FLATBUFFERS_INSTALL_PREFIX}")

if(NOT EXISTS "${FLATBUFFERS_CMAKE_CONFIG_FILE}")
  message(STATUS "Flatbuffers not found. Configuring external build...")

  ExternalProject_Add(
    flatbuffers_external
    DEPENDS
      absl_external
      gtest_external
    GIT_REPOSITORY
      https://github.com/google/flatbuffers.git
    GIT_TAG
      v25.9.23
    PREFIX
      ${FLATBUFFERS_EXT_PREFIX}
    PATCH_COMMAND
      git checkout -- . && git clean -df
    CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=${FLATBUFFERS_INSTALL_PREFIX}
        -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD}
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
        -DFLATBUFFERS_BUILD_TESTS=OFF
        -DFLATBUFFERS_BUILD_GRPCTEST=OFF
        -DFLATBUFFERS_INSTALL=ON
        -DFLATBUFFERS_BUILD_FLATC=ON
        -DFLATBUFFERS_BUILD_FLATHASH=OFF
        -DFLATBUFFERS_CPP_STD=20
    STEP_TARGETS
      step_verify_install
  )
  verify_install(flatbuffers_external ${FLATBUFFERS_CMAKE_CONFIG_FILE})
else()
    message(STATUS "Flatbuffers already installed at: ${FLATBUFFERS_INSTALL_PREFIX}")
    if(NOT TARGET flatbuffers_external)
        add_custom_target(flatbuffers_external)
    endif()
endif()

include(${FLATBUFFERS_PACKAGE_DIR}/flatbuffers_aggregate.cmake)
generate_flatbuffers_aggregate()




set(schema_fbs
  "${PROJECT_ROOT}/schema/core/litertlm_header_schema.fbs"
)
compile_flatbuffer_files(${schema_fbs})