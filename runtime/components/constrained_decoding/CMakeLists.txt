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


set(PKG_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

add_litertlm_library(runtime_components_constrained_decoding_bitmap INTERFACE)
add_library(LiteRTLM::Runtime::Components::ConstrainedDecoding::Bitmap ALIAS runtime_components_constrained_decoding_bitmap)

target_include_directories(runtime_components_constrained_decoding_bitmap
  INTERFACE
    ${PKG_ROOT}
    ${LITERTLM_INCLUDE_PATHS}
)

add_litertlm_library(runtime_components_constrained_decoding_constraint INTERFACE)
add_library(LiteRTLM::Runtime::Components::ConstrainedDecoding::Constraint ALIAS runtime_components_constrained_decoding_constraint)

target_include_directories(runtime_components_constrained_decoding_constraint
  INTERFACE
    ${PKG_ROOT}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_components_constrained_decoding_constraint
  INTERFACE
    LiteRTLM::Runtime::Components::ConstrainedDecoding::Bitmap
    LITERTLM_DEP
)

add_litertlm_library(runtime_components_constrained_decoding_constraint_provider INTERFACE)
add_library(LiteRTLM::Runtime::Components::ConstrainedDecoding::ConstraintProvider ALIAS runtime_components_constrained_decoding_constraint_provider)

target_include_directories(runtime_components_constrained_decoding_constraint_provider
  INTERFACE
    ${PKG_ROOT}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_components_constrained_decoding_constraint_provider
  INTERFACE
    LiteRTLM::Runtime::Components::ConstrainedDecoding::Constraint
    LITERTLM_DEPS
)

add_litertlm_library(runtime_components_constrained_decoding_constrained_decoder STATIC
  constrained_decoder.cc
)
add_library(LiteRTLM::Runtime::Components::ConstrainedDecoding::Decoder ALIAS runtime_components_constrained_decoding_constrained_decoder)

target_include_directories(runtime_components_constrained_decoding_constrained_decoder
  PRIVATE
    ${LITERTLM_INCLUDE_PATHS}
    ${LITERT_INCLUDE_PATHS}
)

target_link_libraries(runtime_components_constrained_decoding_constrained_decoder
  PUBLIC
    LiteRTLM::Runtime::Components::ConstrainedDecoding::Constraint
    LiteRTLM::Runtime::Util::ConvertTensorBuffer
    LiteRTLM::Runtime::Util::LiteRtStatusUtil

    LITERTLM_DEPS
)


add_litertlm_library(runtime_components_constrained_decoding_fake_constraint STATIC
  fake_constraint.cc
)
add_library(LiteRTLM::Runtime::Components::ConstrainedDecoding::FakeConstraint ALIAS runtime_components_constrained_decoding_fake_constraint)

target_include_directories(runtime_components_constrained_decoding_fake_constraint
  PRIVATE
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_components_constrained_decoding_fake_constraint
  PUBLIC
    LiteRTLM::Runtime::Components::ConstrainedDecoding::Bitmap
    LiteRTLM::Runtime::Components::ConstrainedDecoding::Constraint
    LITERTLM_DEPS
)

add_library(runtime_components_constrained_decoding_libs INTERFACE)
add_library(LiteRTLM::Runtime::Components::ConstrainedDecoding ALIAS runtime_components_constrained_decoding_libs)

target_link_libraries(runtime_components_constrained_decoding_libs INTERFACE
  LiteRTLM::Runtime::Components::ConstrainedDecoding::Bitmap
  LiteRTLM::Runtime::Components::ConstrainedDecoding::Constraint
  LiteRTLM::Runtime::Components::ConstrainedDecoding::ConstraintProvider
  LiteRTLM::Runtime::Components::ConstrainedDecoding::Decoder
  LiteRTLM::Runtime::Components::ConstrainedDecoding::FakeConstraint
)