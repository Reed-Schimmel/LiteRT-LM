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


if(NOT TARGET LiteRTLM::absl::absl)
    separate_arguments(_ABSL_LIST NATIVE_COMMAND "${ABSL_LIBS_FLAT}")
    add_library(LiteRTLM::absl::absl INTERFACE IMPORTED GLOBAL)
    set_target_properties(LiteRTLM::absl::absl PROPERTIES
        INTERFACE_LINK_LIBRARIES "-Wl,--start-group;${_ABSL_LIST};-Wl,--end-group"
        INTERFACE_INCLUDE_DIRECTORIES "${ABSL_INCLUDE_DIR}"
    )
endif()

set(ABSL_MOCK_TARGETS
    absl::status
    absl::statusor
    absl::string_view
    absl::strings
    absl::str_format
    absl::log
    absl::check
    absl::flat_hash_map
    absl::flat_hash_set
)

foreach(_target ${ABSL_MOCK_TARGETS})
    if(NOT TARGET ${_target})
        add_library(${_target} INTERFACE IMPORTED GLOBAL)
        set_target_properties(${_target} PROPERTIES
            INTERFACE_LINK_LIBRARIES LiteRTLM::absl::absl
        )
        message(STATUS "[LITERTLM-SHIM] Aliased ${_target} -> LiteRTLM::absl::absl")
    endif()
endforeach()