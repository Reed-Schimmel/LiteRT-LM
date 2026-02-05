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


if(NOT TARGET LiteRTLM::protobuf::libprotobuf)
    add_library(LiteRTLM::protobuf::libprotobuf INTERFACE IMPORTED GLOBAL)
    set_target_properties(LiteRTLM::protobuf::libprotobuf PROPERTIES
        INTERFACE_LINK_LIBRARIES "-Wl,--start-group;${LITERTLM_PROTO_LIBRARIES};-Wl,--end-group"
        INTERFACE_INCLUDE_DIRECTORIES "${LITERTLM_PROTO_INCLUDE_DIRS};${LITERTLM_ABSL_INCLUDE_DIRS}"
    )

    if(NOT TARGET protobuf::libprotobuf)
        add_library(protobuf::libprotobuf ALIAS LiteRTLM::protobuf::libprotobuf)
    endif()
    if(NOT TARGET protobuf::protobuf)
        add_library(protobuf::protobuf ALIAS LiteRTLM::protobuf::libprotobuf)
    endif()

    set(Protobuf_INCLUDE_DIR "${LITERTLM_PROTO_INCLUDE_DIRS}" CACHE INTERNAL "")
    set(Protobuf_LIBRARIES LiteRTLM::protobuf::libprotobuf CACHE INTERNAL "")
    set(Protobuf_PROTOC_EXECUTABLE "${LITERTLM_PROTOC_EXECUTABLE}" CACHE INTERNAL "")
    set(Protobuf_FOUND TRUE CACHE INTERNAL "")
    set(PROTOBUF_FOUND TRUE CACHE INTERNAL "")
endif()