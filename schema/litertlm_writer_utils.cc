
#include "schema/litertlm_writer_utils.h"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <ios>
#include <iostream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/numbers.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_split.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "third_party/flatbuffers/include/flatbuffers/flatbuffer_builder.h"
#include "runtime/proto/llm_metadata.pb.h"
#include "schema/core/litertlm_export.h"
#include "schema/core/litertlm_header.h"
#include "schema/core/litertlm_header_schema_generated.h"
#include "schema/core/litertlm_section.h"
#include "google/protobuf/text_format.h"  // from @com_google_protobuf

namespace litert::litertlm::schema {

// Section names used in the section_metadata flag.
constexpr char kTokenizerSectionName[] = "tokenizer";
constexpr char kTfliteSectionName[] = "tflite";
constexpr char kLlmMetadataSectionName[] = "llm_metadata";

using ::litert::lm::proto::LlmMetadata;

// Helper function to parse a single key-value pair.
absl::Status ParseKeyValuePair(absl::string_view kv_str, std::string& key,
                               std::string& value) {
  std::vector<std::string> parts = absl::StrSplit(kv_str, '=');
  if (parts.size() != 2) {
    return absl::InvalidArgumentError(
        absl::StrCat("Invalid key-value pair: ", kv_str));
  }
  key = parts[0];
  value = parts[1];
  return absl::OkStatus();
}

// Helper function to convert string value to the correct type.
// This function returns a KVPair object.
KVPair ConvertKeyValue(flatbuffers::FlatBufferBuilder& builder,
                       const std::string& key, const std::string& value_str) {
  int32_t int32_value;
  int64_t int64_value;
  uint32_t uint32_value;
  uint64_t uint64_value;
  float float_value;
  bool bool_value;

  if (absl::SimpleAtoi(value_str, &int32_value)) {
    return CreateKeyValuePair(builder, key, int32_value);
  } else if (absl::SimpleAtoi(value_str, &int64_value)) {
    return CreateKeyValuePair(builder, key, int64_value);
  } else if (absl::SimpleAtoi(value_str, &uint32_value)) {
    return CreateKeyValuePair(builder, key, uint32_value);
  } else if (absl::SimpleAtoi(value_str, &uint64_value)) {
    return CreateKeyValuePair(builder, key, uint64_value);
  } else if (absl::SimpleAtof(value_str, &float_value)) {
    return CreateKeyValuePair(builder, key, float_value);
  } else if (value_str == "true" || value_str == "false") {
    bool_value = (value_str == "true");
    return CreateKeyValuePair(builder, key, bool_value);
  } else {
    // Default to string.
    return CreateKeyValuePair(builder, key, value_str);
  }
}

// Helper function to get file extension
std::string GetFileExtension(const std::string& filename) {
  size_t dot_pos = filename.rfind('.');
  if (dot_pos == std::string::npos) {
    return "";
  }
  return filename.substr(dot_pos);
}

absl::Status LitertLmWrite(const std::vector<std::string>& command_args,
                           const std::string& section_metadata_str,
                           const std::string& output_path) {
  std::vector<std::unique_ptr<SectionStreamBase>> sections;
  std::vector<AnySectionDataType> section_types;
  // To store the order of section names derived from input filenames.
  std::vector<std::string> section_name_order;

  if (command_args.empty()) {
    return absl::InvalidArgumentError(
        "At least one input file must be provided.");
  }

  for (const auto& filename : command_args) {
    std::string extension = GetFileExtension(filename);
    ABSL_LOG(INFO) << "Processing file: " << filename
                   << " with extension: " << extension;

    if (extension == ".tflite") {
      sections.push_back(std::make_unique<FileBackedSectionStream>(filename));
      section_types.push_back(AnySectionDataType_TFLiteModel);
      section_name_order.push_back(kTfliteSectionName);
    } else if (extension == ".pb" || extension == ".proto") {
      LlmMetadata llm_metadata_proto;
      std::ifstream ifs(filename, std::ios::binary);
      if (!ifs.is_open()) {
        return absl::NotFoundError(absl::StrCat(
            "Could not open llm_metadata binary file: ", filename));
      }
      std::string proto_str((std::istreambuf_iterator<char>(ifs)),
                            std::istreambuf_iterator<char>());
      if (!llm_metadata_proto.ParseFromString(proto_str)) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Failed to parse LlmMetadata protobuf from binary file: ",
            filename));
      }
      sections.push_back(std::make_unique<ProtoBufSectionStream<LlmMetadata>>(
          llm_metadata_proto));
      section_types.push_back(AnySectionDataType_LlmMetadataProto);
      section_name_order.push_back(kLlmMetadataSectionName);
    } else if (extension == ".pbtext" || extension == ".prototext") {
      LlmMetadata llm_metadata_proto;
      std::ifstream ifs(filename);
      if (!ifs.is_open()) {
        return absl::NotFoundError(
            absl::StrCat("Could not open llm_metadata text file: ", filename));
      }
      std::string proto_text_str((std::istreambuf_iterator<char>(ifs)),
                                 std::istreambuf_iterator<char>());
      if (!proto2::TextFormat::ParseFromString(proto_text_str,
                                               &llm_metadata_proto)) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Failed to parse LlmMetadata protobuf from text file: ", filename));
      }
      sections.push_back(std::make_unique<ProtoBufSectionStream<LlmMetadata>>(
          llm_metadata_proto));
      section_types.push_back(AnySectionDataType_LlmMetadataProto);
      section_name_order.push_back(kLlmMetadataSectionName);
    } else if (extension == ".spiece") {
      sections.push_back(std::make_unique<FileBackedSectionStream>(filename));
      section_types.push_back(AnySectionDataType_SP_Tokenizer);
      section_name_order.push_back(kTokenizerSectionName);
    } else {
      return absl::InvalidArgumentError(
          absl::StrCat("Unsupported file extension for: ", filename,
                       ". Supported extensions: .tflite, .pb, .proto, .pbtext, "
                       ".prototext, .spiece"));
    }
  }

  if (sections.empty()) {
    return absl::InvalidArgumentError("No valid input files provided.");
  }

  flatbuffers::FlatBufferBuilder builder;
  std::vector<std::vector<KVPair>> section_items_list(sections.size());
  std::vector<std::string> metadata_section_order;

  if (!section_metadata_str.empty()) {
    std::vector<std::string> section_parts =
        absl::StrSplit(section_metadata_str, ';');
    for (const auto& section_part : section_parts) {
      std::vector<std::string> parts = absl::StrSplit(section_part, ':');
      if (parts.size() != 2) {
        return absl::InvalidArgumentError(
            absl::StrCat("Invalid section metadata format: ", section_part,
                         ". Expected 'section_name:key1=value1,...'"));
      }
      std::string section_name = parts[0];
      metadata_section_order.push_back(section_name);
    }
  }

  if (section_name_order.size() != metadata_section_order.size() &&
      !section_metadata_str.empty()) {  // Only check if metadata is provided
    return absl::InvalidArgumentError(
        absl::StrCat("Mismatch in number of sections between input files (",
                     section_name_order.size(), ") and section_metadata (",
                     metadata_section_order.size(),
                     "). "
                     "The number of sections provided via filenames must match "
                     "the number of sections "
                     "defined in the --section_metadata flag."));
  }

  // Only perform order check if metadata is actually provided.
  if (!section_metadata_str.empty()) {
    for (size_t i = 0; i < section_name_order.size(); ++i) {
      if (section_name_order[i] != metadata_section_order[i]) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Order mismatch for section at index ", i,
            ". Expected section from filename: '", section_name_order[i],
            "', Found in metadata: '", metadata_section_order[i],
            "'. The order of sections in --section_metadata must match the "
            "order "
            "of input filenames."));
      }
    }
  }

  if (!section_metadata_str.empty()) {
    int current_metadata_section_index = 0;
    std::vector<std::string> section_parts =
        absl::StrSplit(section_metadata_str, ';');
    for (const auto& section_part : section_parts) {
      std::vector<std::string> parts = absl::StrSplit(section_part, ':');
      // section_name = parts[0]; // Already validated or not needed here due to
      // order enforcement
      std::vector<std::string> kv_pairs_str = absl::StrSplit(parts[1], ',');
      for (const auto& kv_str : kv_pairs_str) {
        std::string key, value_str;
        absl::Status parsed_status = ParseKeyValuePair(kv_str, key, value_str);
        if (!parsed_status.ok()) {
          return absl::InvalidArgumentError(absl::StrCat(
              "Failed to parse key-value pair '", kv_str, "' in section '",
              parts[0], "': ", parsed_status.message()));
        }
        section_items_list[current_metadata_section_index].push_back(
            ConvertKeyValue(builder, key, value_str));
      }
      current_metadata_section_index++;
    }
  }

  std::vector<KVPair> system_meta = {
      CreateKeyValuePair(
          builder, std::string("arch"),
          CreateStringValue(builder, builder.CreateString(std::string("all")))),
      CreateKeyValuePair(builder, std::string("version"),
                         CreateStringValue(builder, builder.CreateString(
                                                        std::string("0.1"))))};

  return MakeLiteRTLMFromSections(builder, sections, section_types, system_meta,
                                  section_items_list, output_path);
}

}  // namespace litert::litertlm::schema
