#include "schema/core/litertlm_read.h"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iosfwd>
#include <iostream>
#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "schema/core/litertlm_header_schema_generated.h"
#include "schema/core/litertlm_utils.h"
#include "sentencepiece_processor.h"  // from @sentencepiece
#include "tensorflow/lite/model_builder.h"  // from @org_tensorflow

namespace litert {
namespace litertlm {
namespace schema {

using litert::lm::proto::LlmMetadata;

absl::Status ReadHeaderFromLiteRTLM(std::istream& litertlm_stream,
                                    LitertlmHeader* header, int* major_version,
                                    int* minor_version, int* patch_version) {
  // 0. Read magic number and version.
  char magic_number[8];
  litertlm_stream.read(magic_number, 8);
  if (litertlm_stream.gcount() != 8 ||
      std::string(magic_number, 8) != "LITERTLM") {
    return absl::InvalidArgumentError(
        absl::StrFormat("Invalid magic number or failed to read: %s",
                        std::string(magic_number, litertlm_stream.gcount())));
  }

  uint8_t major_version_u8, minor_version_u8, patch_version_u8;
  litertlm_stream.read(reinterpret_cast<char*>(&major_version_u8),
                       sizeof(uint8_t));
  litertlm_stream.read(reinterpret_cast<char*>(&minor_version_u8),
                       sizeof(uint8_t));
  litertlm_stream.read(reinterpret_cast<char*>(&patch_version_u8),
                       sizeof(uint8_t));

  if (!litertlm_stream) {
    return absl::InternalError("Failed to read version bytes.");
  }

  // Store the version as ints.
  *major_version = static_cast<int>(major_version_u8);
  *minor_version = static_cast<int>(minor_version_u8);
  *patch_version = static_cast<int>(patch_version_u8);

  // TODO(talumbau) Assert that major version matches current major version
  // constant.

  // 1. Skip 5 bytes of padding.
  litertlm_stream.ignore(5);
  if (!litertlm_stream) {
    return absl::InternalError("Failed to skip padding after version.");
  }

  // 2. Read the header end offset.
  uint64_t header_end_offset;
  litertlm_stream.read(reinterpret_cast<char*>(&header_end_offset),
                       sizeof(uint64_t));
  if (!litertlm_stream) {
    return absl::InternalError("Failed to read header end offset.");
  }

  // 3. Skip 8 bytes of padding.
  litertlm_stream.ignore(8);
  if (!litertlm_stream) {
    return absl::InternalError(
        "Failed to skip padding after header end offset.");
  }

  // Calculate the header size.
  std::streampos current_position = litertlm_stream.tellg();
  if (current_position == -1) {
    return absl::InternalError("Failed to get current stream position.");
  }
  // Ensure header_end_offset is greater than or equal to current_position
  if (header_end_offset < static_cast<uint64_t>(current_position)) {
    return absl::InvalidArgumentError(
        "Invalid header end offset: smaller than current position.");
  }
  uint64_t header_size =
      header_end_offset - static_cast<uint64_t>(current_position);

  // 3. Read the header data into a buffer.
  auto header_buffer = std::make_unique<uint8_t[]>(header_size);

  litertlm_stream.read(reinterpret_cast<char*>(header_buffer.get()),
                       header_size);
  if (!litertlm_stream) {
    return absl::InternalError("Failed to read header data.");
  }

  header->reset(std::move(header_buffer));
  return absl::OkStatus();
}

// The public function that takes a file path.
absl::Status ReadHeaderFromLiteRTLM(const std::string& litertlm_path,
                                    LitertlmHeader* header, int* major_version,
                                    int* minor_version, int* patch_version) {
  std::ifstream input_file_stream(litertlm_path, std::ios::binary);
  if (!input_file_stream.is_open()) {
    return absl::InternalError(
        absl::StrFormat("Could not open file: %s", litertlm_path));
  }

  absl::Status status = ReadHeaderFromLiteRTLM(
      input_file_stream, header, major_version, minor_version, patch_version);

  return status;
}

// The public function that takes a pointer and a length.
absl::Status ReadHeaderFromLiteRTLM(void* data, std::size_t length,
                                    LitertlmHeader* header, int* major_version,
                                    int* minor_version, int* patch_version) {
  char* char_data = static_cast<char*>(data);
  // Create a streambuf instance based on the given buffer info.
  MemoryStreamBuf sbuf(char_data, length);
  // Create an istream using the custom streambuf.
  std::istream input_stream(&sbuf);

  absl::Status status = ReadHeaderFromLiteRTLM(
      input_stream, header, major_version, minor_version, patch_version);
  // Cleanup of the streambuf and istream is automatic upon exit.
  return status;
}

template <AnySectionDataType SectionT, typename T>
absl::Status ReadValueTFromSection(
    const std::string& litertlm_path, int section_idx, T* data,
    std::function<absl::Status(std::ifstream&, uint64_t, uint64_t, T*)>
        read_section_into_t) {
  LitertlmHeader header;
  int major_version, minor_version, patch_version;

  // Read the header information.
  RETURN_IF_ERROR(ReadHeaderFromLiteRTLM(
      litertlm_path, &header, &major_version, &minor_version, &patch_version));

  auto sections = header.metadata->section_metadata()->objects();
  // Check if the section_idx is valid.
  if (section_idx < 0 || section_idx >= sections->size()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Invalid section index: %d, num sections = %d",
                        section_idx, sections->size()));
  }

  const SectionObject* section = sections->Get(section_idx);

  // Verify that the section type is correct.
  if (section->data_type() != SectionT) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Section %d is not the expected type.  It is: %s, expected %d",
        section_idx, AnySectionDataTypeToString(section->data_type()),
        static_cast<int>(SectionT)));
  }

  // Calculate the size of the data.
  size_t end_offset = section->end_offset();
  size_t begin_offset = section->begin_offset();
  size_t data_size = end_offset - begin_offset;
  if (data_size == 0) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Section %d has zero size.", section_idx));
  }

  // Read the data from the file using the provided function.
  std::ifstream input_file_stream(litertlm_path, std::ios::binary);
  if (!input_file_stream.is_open()) {
    return absl::InternalError(
        absl::StrFormat("Could not open file: %s", litertlm_path));
  }
  input_file_stream.seekg(section->begin_offset());

  return read_section_into_t(input_file_stream, begin_offset, end_offset, data);
}

// Function to read TFLite model data from a section.
absl::Status ReadSectionIntoTFLite(
    std::ifstream& input_stream, uint64_t begin_offset, uint64_t end_offset,
    std::unique_ptr<tflite::FlatBufferModel>* tflite_model) {
  size_t model_size = end_offset - begin_offset;
  std::unique_ptr<char[]> buffer(new char[model_size]);
  input_stream.read(buffer.get(), model_size);
  if (!input_stream) {
    return absl::InternalError(
        absl::StrFormat("Could not read %d bytes from stream.", model_size));
  }

  // Store the model in the output parameter.
  *tflite_model =
      tflite::FlatBufferModel::BuildFromBuffer(buffer.get(), model_size);
  return absl::OkStatus();
}

// Function to read LlmMetadata from a section.
absl::Status ReadSectionIntoLlmMetadata(std::ifstream& input_stream,
                                        uint64_t begin_offset,
                                        uint64_t end_offset,
                                        LlmMetadata* llm_metadata) {
  size_t size = end_offset - begin_offset;
  std::unique_ptr<char[]> buffer(new char[size]);
  input_stream.read(buffer.get(), size);
  if (!input_stream) {
    return absl::InternalError(
        absl::StrFormat("Could not read %d bytes from stream.", size));
  }
  llm_metadata->ParseFromArray(buffer.get(), size);
  return absl::OkStatus();
}

// Function to read a SP Tokenizer from a section.
absl::Status ReadSectionIntoSPTokenizer(
    std::ifstream& input_stream, uint64_t begin_offset, uint64_t end_offset,
    sentencepiece::SentencePieceProcessor* sp_proc) {
  size_t size = end_offset - begin_offset;
  std::unique_ptr<char[]> buffer(new char[size]);
  input_stream.read(buffer.get(), size);
  if (!input_stream) {
    return absl::InternalError(
        absl::StrFormat("Could not read %d bytes from stream.", size));
  }
  absl::string_view buffer_view(buffer.get(), size);
  return sp_proc->LoadFromSerializedProto(buffer_view);
}

absl::Status ReadTFLiteFromSection(
    const std::string& litertlm_path, int section_idx,
    std::unique_ptr<tflite::FlatBufferModel>* tflite_model) {
  return ReadValueTFromSection<AnySectionDataType_TFLiteModel,
                               std::unique_ptr<tflite::FlatBufferModel>>(
      litertlm_path, section_idx, tflite_model, ReadSectionIntoTFLite);
}

absl::Status ReadLlmMetadataFromSection(const std::string& litertlm_path,
                                        int section_idx,
                                        LlmMetadata* llm_metadata) {
  return ReadValueTFromSection<AnySectionDataType_LlmMetadataProto,
                               LlmMetadata>(
      litertlm_path, section_idx, llm_metadata, ReadSectionIntoLlmMetadata);
}

absl::Status ReadSPTokenizerFromSection(
    const std::string& litertlm_path, int section_idx,
    sentencepiece::SentencePieceProcessor* sp_proc) {
  return ReadValueTFromSection<AnySectionDataType_SP_Tokenizer,
                               sentencepiece::SentencePieceProcessor>(
      litertlm_path, section_idx, sp_proc, ReadSectionIntoSPTokenizer);
}

template <AnySectionDataType SectionT, typename T>
absl::Status ReadAnyT(const std::string& litertlm_path, T* data,
                      std::function<absl::Status(const std::string&, int, T*)>
                          read_data_from_section) {
  LitertlmHeader header;
  int major_version, minor_version, patch_version;

  // Read the header information.
  RETURN_IF_ERROR(ReadHeaderFromLiteRTLM(
      litertlm_path, &header, &major_version, &minor_version, &patch_version));

  // Search for the first section with the specified type.
  auto sections = header.metadata->section_metadata()->objects();
  int section_index = -1;
  for (size_t i = 0; i < sections->size(); ++i) {
    const SectionObject* section = sections->Get(i);
    if (section->data_type() == SectionT) {
      section_index = static_cast<int>(i);
      break;
    }
  }

  if (section_index == -1) {
    return absl::NotFoundError("No matching section found in the file.");
  }

  // Read the data from the found section.
  return read_data_from_section(litertlm_path, section_index, data);
}

// Instantiation of ReadAnyT for TFLite models.
absl::Status ReadAnyTFLite(
    const std::string& litertlm_path,
    std::unique_ptr<tflite::FlatBufferModel>* tflite_model) {
  return ReadAnyT<AnySectionDataType_TFLiteModel,
                  std::unique_ptr<tflite::FlatBufferModel>>(
      litertlm_path, tflite_model, ReadTFLiteFromSection);
}

// Instantiation of ReadAnyT for LlmMetadata.
absl::Status ReadAnyLlmMetadata(const std::string& litertlm_path,
                                LlmMetadata* llm_metadata) {
  return ReadAnyT<AnySectionDataType_LlmMetadataProto, LlmMetadata>(
      litertlm_path, llm_metadata, ReadLlmMetadataFromSection);
}

// Instantiation of ReadAnyT for LlmMetadata.
absl::Status ReadAnySPTokenizer(
    const std::string& litertlm_path,
    sentencepiece::SentencePieceProcessor* sp_proc) {
  return ReadAnyT<AnySectionDataType_SP_Tokenizer,
                  sentencepiece::SentencePieceProcessor>(
      litertlm_path, sp_proc, ReadSPTokenizerFromSection);
}

}  // namespace schema
}  // namespace litertlm
}  // namespace litert
