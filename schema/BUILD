# Copyright 2025 The ODML Authors.
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

package_group(
    name = "litert_lm_schema",
    packages = [
        "//schema/...",
    ],
)

package(
    default_applicable_licenses = ["//third_party/odml:license"],
    default_compatible_with = [
        "//buildenv/target:non_prod",
        "//buildenv/target:prod",
    ],
    default_hdrs_check = "strict",
    default_visibility = [
        "//third_party/odml:__subpackages__",
    ],
)

licenses(["notice"])

filegroup(
    name = "testdata",
    srcs = glob(["testdata/*"]),
)

cc_binary(
    name = "litertlm_writer",
    srcs = ["litertlm_writer.cc"],
    deps = [
        ":litertlm_writer_utils",
        "//base",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "//third_party/flatbuffers",
        "//runtime/proto:llm_metadata_cc_proto",
        "//schema/core:litertlm_export",
        "//schema/core:litertlm_header",
        "//schema/core:litertlm_header_schema",
        "//schema/core:litertlm_section",
        "@com_google_protobuf//:protobuf",
    ],
)

cc_binary(
    name = "litertlm_export_main",
    srcs = ["litertlm_export_main.cc"],
    deps = [
        "//base",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "//third_party/flatbuffers",
        "//runtime/proto:llm_metadata_cc_proto",
        "//schema/core:litertlm_export",
        "//schema/core:litertlm_header",
        "//schema/core:litertlm_header_schema",
        "//schema/core:litertlm_section",
        "@com_google_protobuf//:protobuf",
    ],
)

cc_binary(
    name = "litertlm_peek",
    srcs = ["litertlm_peek.cc"],
    deps = [
        "//base",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "//schema/core:litertlm_header",
        "//schema/core:litertlm_header_schema",
        "//schema/core:litertlm_print",
        "//schema/core:litertlm_read",
        "//schema/core:litertlm_section",
    ],
)

cc_test(
    name = "litertlm_writer_test",
    srcs = ["litertlm_writer_test.cc"],
    data = [
        "//schema:testdata",
    ],
    deps = [
        ":litertlm_writer_utils",
        "//devtools/build/runtime:get_runfiles_dir",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "//third_party/flatbuffers",
        "//runtime/proto:llm_metadata_cc_proto",
        "//schema/core:litertlm_print",
        "@com_google_protobuf//:protobuf",
        "@sentencepiece//:sentencepiece_processor",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite:framework_stable",
        "@org_tensorflow//tensorflow/lite/c:common",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
        "@org_tensorflow//tensorflow/lite/schema:schema_fbs",
    ],
)

cc_library(
    name = "litertlm_writer_utils",
    srcs = ["litertlm_writer_utils.cc"],
    hdrs = ["litertlm_writer_utils.h"],
    deps = [
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "//third_party/flatbuffers",
        "//runtime/proto:llm_metadata_cc_proto",
        "//schema/core:litertlm_export",
        "//schema/core:litertlm_header",
        "//schema/core:litertlm_header_schema",
        "//schema/core:litertlm_print",
        "//schema/core:litertlm_section",
        "//schema/core:litertlm_utils",
        "@com_google_protobuf//:protobuf",
    ],
)
