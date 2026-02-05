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

set(TFLITE_EXT_PREFIX       "${EXTERNAL_PROJECT_BINARY_DIR}/tensorflow" CACHE INTERNAL "")
set(TFLITE_INSTALL_PREFIX   "${TFLITE_EXT_PREFIX}/install" CACHE INTERNAL "")
set(TFLITE_INCLUDE_DIR      "${TFLITE_INSTALL_PREFIX}/include" CACHE INTERNAL "")
set(TFLITE_LIB_DIR          "${TFLITE_INSTALL_PREFIX}/lib" CACHE INTERNAL "")
set(TFLITE_SRC_DIR          "${TFLITE_EXT_PREFIX}/src/tflite_external/tensorflow/lite" CACHE INTERNAL "")
set(TFLITE_BUILD_DIR        "${TFLITE_EXT_PREFIX}/src/tflite_external-build" CACHE INTERNAL "TFLite Build Directory")
set(TENSORFLOW_SOURCE_DIR   "${TFLITE_EXT_PREFIX}/src/tflite_external" CACHE INTERNAL "")
set(TFLITE_STATIC_LIB       "${TFLITE_BUILD_DIR}/libtensorflow-lite.a" CACHE INTERNAL "")
set(RUY_INCLUDE_DIR         "${EXTERNAL_PROJECT_BINARY_DIR}/tflite_external-build" CACHE INTERNAL "")

setup_external_install_structure("${TFLITE_INSTALL_PREFIX}")

if(NOT EXISTS "${TFLITE_STATIC_LIB}")
    message(STATUS "TFLite not found. Configuring external build...")

    set(SHIM_CODE
        "add_library(LiteRTLM::absl::absl INTERFACE IMPORTED GLOBAL)
         set_target_properties(LiteRTLM::absl::absl PROPERTIES INTERFACE_LINK_LIBRARIES \"-Wl,--start-group ${ABSL_LIBS_FLAT} -Wl,--end-group\")"
    )

    ExternalProject_Add(
        tflite_external
        DEPENDS
            absl_external
            flatbuffers_external
            gtest_external
            opencl_headers_external
            protobuf_external
            tokenizers-cpp_external
        GIT_REPOSITORY  https://github.com/tensorflow/tensorflow.git
        GIT_TAG         061041963ead867e8f47fb63e153db3e61e3b20b
        PREFIX          "${TFLITE_EXT_PREFIX}"
        SOURCE_SUBDIR   "tensorflow/lite"

        PATCH_COMMAND
            git checkout -- . && git clean -df

            COMMAND ${CMAKE_COMMAND}
                -DFLATC_EXECUTABLE=${FLATC_EXECUTABLE}
                -DTFLITE_SRC_DIR=${TFLITE_SRC_DIR}
                -DTFLITE_BUILD_DIR=${TFLITE_BUILD_DIR}
                -DTENSORFLOW_SOURCE_DIR=${TENSORFLOW_SOURCE_DIR}
                -DLITERTLM_PACKAGES_DIR=${LITERTLM_PACKAGES_DIR}
                -DLITERTLM_MODULES_DIR=${LITERTLM_MODULES_DIR}
                -DTFLITE_PACKAGE_DIR=${TFLITE_PACKAGE_DIR}
                -DABSL_INCLUDE_DIR=${ABSL_INCLUDE_DIR}
                -DPROTOBUF_PACKAGE_DIR=${PROTOBUF_PACKAGE_DIR}
                -DPROTOBUF_INCLUDE_DIR=${PROTOBUF_INCLUDE_DIR}
                -DLITERTLM_PATCHES_DIR=${LITERTLM_PATCHES_DIR}
                -DLITERTLM_PROJECT_ROOT=${PROJECT_ROOT}
                -DFLATBUFFERS_INCLUDE_DIR=${FLATBUFFERS_INCLUDE_DIR}
                -P "${LITERTLM_PACKAGES_DIR}/tflite/tflite_patcher.cmake"

        CMAKE_ARGS
            -DCMAKE_INSTALL_PREFIX=${TFLITE_INSTALL_PREFIX}
            -DCMAKE_POLICY_VERSION_MINIMUM=3.5
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            -DCMAKE_POLICY_DEFAULT_CMP0169=OLD
            -DCMAKE_POLICY_DEFAULT_CMP0170=OLD
            -DCMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD}
            -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
            -DCMAKE_POSITION_INDEPENDENT_CODE=ON
            -DCMAKE_CROSSCOMPILING=ON
            "-DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS} -DTF_MAJOR_VERSION=2 -DTF_MINOR_VERSION=20 -DTF_PATCH_VERSION=0 -DTF_VERSION_SUFFIX=\"\" -w"
            "-DCMAKE_C_FLAGS=${CMAKE_C_FLAGS} -DTF_MAJOR_VERSION=2 -DTF_MINOR_VERSION=20 -DTF_PATCH_VERSION=0 -DTF_VERSION_SUFFIX=\"\""

            -DTFLITE_HOST_TOOLS_DIR=${LITERTLM_INSTALL_DIR}/bin

            -Dabsl_DIR=${ABSL_INSTALL_PREFIX}/lib/cmake/absl
            -D_abseil-cpp_LICENSE_FILE:FILEPATH=${ABSL_SRC_DIR}/absl_external/LICENSE
            -DLITERTLM_ABSL_LIBRARIES=${ABSL_LIBS_FLAT}
            -DABSL_INCLUDE_DIR=${ABSL_INCLUDE_DIR}
            -DPROTOBUF_INCLUDE_DIR=${PROTOBUF_INCLUDE_DIR}

            -DFLATBUFFERS_INCLUDE_DIR=${FLATBUFFERS_INCLUDE_DIR}
            -DFLATBUFFERS_BUILD_FLATC=OFF
            -DFLATBUFFERS_INSTALL=OFF
            "-DFlatBuffers_BINARY_DIR=${FLATBUFFERS_BIN_DIR}"
            "-DFLATBUFFERS_PROJECT_DIR=${FLATBUFFERS_SRC_DIR}/flatbuffers_external"
            "-DFlatBuffers_SOURCE_DIR=${FLATBUFFERS_SRC_DIR}/flatbuffers_external"
            "-D_flatbuffers_LICENSE_FILE=${FLATBUFFERS_SRC_DIR}/flatbuffers_external/LICENSE"
            "-DFLATC_PATHS=${FLATBUFFERS_BIN_DIR}"
            "-DFLATBUFFERS_FLATC_EXECUTABLE=${FLATC_EXECUTABLE}"
            "-Dflatbuffers_DIR=${FLATBUFFERS_INSTALL_PREFIX}/lib/cmake/flatbuffers"
            "-DFLATC_TARGET=${FLATC_EXECUTABLE}"
            "-DFLATC_EXECUTABLE=${FLATC_EXECUTABLE}"
            -DFLATBUFFERS_PACKAGE_DIR=${FLATBUFFERS_PACKAGE_DIR}

            -Dprotobuf_BINARY_DIR=${PROTO_BIN_DIR}
            -Dprotobuf_BUILD_PROTOC_BINARIES=OFF
            -Dprotobuf_SOURCE_DIR=${PROTO_SRC_DIR}
            "-DLITERTLM_PROTO_LIBRARIES=${PROTO_LIBS_FLAT}"
            "-DLITERTLM_PROTO_INCLUDE_DIRS=${PROTO_INCLUDE_DIR}"
            "-DLITERTLM_PROTOC_EXECUTABLE=${PROTO_PROTOC_EXECUTABLE}"

            "-DCMAKE_PREFIX_PATH=${ABSL_INSTALL_PREFIX};${libpng_lib_BINARY_DIR}"
            -DPNG_FOUND=ON
            -DPNG_LIBRARY=${libpng_lib_BINARY_DIR}/libpng.a
            -DPNG_PNG_INCLUDE_DIR=${libpng_lib_SOURCE_DIR}

            -DTFLITE_ENABLE_INSTALL=OFF
            -DTFLITE_ENABLE_XNNPACK=ON
            -DTFLITE_ENABLE_RESOURCE_VARIABLE=OFF
            -DXNNPACK_SET_VERBOSITY=OFF
            -DTFLITE_ENABLE_GPU=OFF
            -DTENSORFLOW_SOURCE_DIR=${TENSORFLOW_SOURCE_DIR}
            "-DTFLITE_HOST_TOOLS_DIR=${FLATBUFFERS_BIN_DIR}"
            "-DCMAKE_EXE_LINKER_FLAGS=-L${ABSL_LIB_DIR} -L${PROTO_INSTALL_PREFIX}/lib"

            "-DCMAKE_CXX_STANDARD_LIBRARIES=-lpthread"
            "-DLITERTLM_PACKAGES_DIR=${LITERTLM_PACKAGES_DIR}"
            -DTFLITE_PACKAGE_DIR=${TFLITE_PACKAGE_DIR}
            -DABSL_PACKAGE_DIR=${ABSL_PACKAGE_DIR}
            -DPROTOBUF_PACKAGE_DIR=${PROTOBUF_PACKAGE_DIR}

            -DLITERTLM_MODULES_DIR=${LITERTLM_MODULES_DIR}
            -DLITERTLM_PROJECT_ROOT=${PROJECT_ROOT}

            -DTFLITE_SRC_DIR=${TFLITE_SRC_DIR}
            -DTFLITE_BUILD_DIR=${TFLITE_BUILD_DIR}
            -DPROTO_PROTOC_EXECUTABLE=${PROTO_PROTOC_EXECUTABLE}
    )

else()
    message(STATUS "[TFLite] Found existing installation at: ${TFLITE_STATIC_LIB}")
    if(NOT TARGET tflite_external)
        add_custom_target(tflite_external)
    endif()
endif()

include(${TFLITE_PACKAGE_DIR}/tflite_aggregate.cmake)
generate_tflite_aggregate()