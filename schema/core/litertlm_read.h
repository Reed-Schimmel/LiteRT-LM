#ifndef THIRD_PARTY_ODML_LITERT_LM_SCHEMA_CORE_LITERTLM_READ_H_
#define THIRD_PARTY_ODML_LITERT_LM_SCHEMA_CORE_LITERTLM_READ_H_

#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "runtime/proto/llm_metadata.pb.h"
#include "schema/core/litertlm_header.h"
#include "schema/core/litertlm_header_schema_generated.h"
#include "schema/core/litertlm_utils.h"
#include "sentencepiece_processor.h"  // from @sentencepiece
#include "tensorflow/lite/model_builder.h"  // from @org_tensorflow
#include "tensorflow/lite/schema/schema_generated.h"  // from @org_tensorflow

namespace litert {

namespace litertlm {

namespace schema {

using litert::lm::proto::LlmMetadata;

struct LitertlmHeader {
  std::unique_ptr<uint8_t[]> buffer;
  const LiteRTLMMetaData* metadata;

  // Default constructor
  LitertlmHeader() : buffer(nullptr), metadata(nullptr) {}

  // Constructor that takes ownership of the buffer.
  LitertlmHeader(std::unique_ptr<uint8_t[]>&& buffer_)
      : buffer(std::move(buffer_)) {
    reset(std::move(buffer_));
  }

  // Disable copy constructor and assignment operator to prevent incorrect
  // copying.
  LitertlmHeader(const LitertlmHeader&) = delete;
  LitertlmHeader& operator=(const LitertlmHeader&) = delete;

  // Add a move constructor and move assignment operator for proper transfer of
  // ownership
  LitertlmHeader(LitertlmHeader&& other) noexcept
      : buffer(std::move(other.buffer)), metadata(other.metadata) {
    other.metadata = nullptr;
  }

  // reset function
  void reset(std::unique_ptr<uint8_t[]>&& buffer_) {
    buffer = std::move(buffer_);
    if (buffer) {
      metadata = GetLiteRTLMMetaData(buffer.get());
    } else {
      metadata = nullptr;
    }
  }

  ~LitertlmHeader() {
    // No need to delete, unique_ptr handles it.
  }
};

// Reads the LiteRTLM file header starting at `data`. It is assumed
// that this function can read up to `length` bytes starting at `data`.
//
// Args:
//   data: The pointer to some buffer we can read from.
//   length: The number of bytes that it is valid to read from starting at
//           `data`
//   header: The LitertlmHeader struct with populated schema data.
//   major_version: The major version of the LiteRTLM file.
//   minor_version: The minor version of the LiteRTLM file.
//   patch_version: The patch version of the LiteRTLM file.
//
// Returns:
//   absl::OkStatus() if the file was read successfully, or an error status
//   otherwise.
absl::Status ReadHeaderFromLiteRTLM(void* data, size_t length,
                                    LitertlmHeader* header, int* major_version,
                                    int* minor_version, int* patch_version);

// Reads the LiteRTLM file from the given path and populates a header
// data structure (allocating and owning data for the header).
//
// Args:
//   litertlm_path: The path to the LiteRTLM file.
//   header: The LitertlmHeader struct with populated schema data.
//   major_version: The major version of the LiteRTLM file.
//   minor_version: The minor version of the LiteRTLM file.
//   patch_version: The patch version of the LiteRTLM file.
//
// Returns:
//   absl::OkStatus() if the file was read successfully, or an error status
//   otherwise.
absl::Status ReadHeaderFromLiteRTLM(const std::string& litertlm_path,
                                    LitertlmHeader* header, int* major_version,
                                    int* minor_version, int* patch_version);

// Reads the LiteRTLM file from the given istream and populates a header
// data structure (allocating and owning data for the header).
//
// Args:
//   litertlm_stream: The input stream to the LiteRTLM file.
//   header: The LitertlmHeader struct with populated schema data.
//   major_version: The major version of the LiteRTLM file.
//   minor_version: The minor version of the LiteRTLM file.
//   patch_version: The patch version of the LiteRTLM file.
//
// Returns:
//   absl::OkStatus() if the file was read successfully, or an error status
//   otherwise.
absl::Status ReadHeaderFromLiteRTLM(std::istream& litertlm_stream,
                                    LitertlmHeader* header, int* major_version,
                                    int* minor_version, int* patch_version);

// Read a TF Lite from the specified section in the LiteRT-LM file.
// Returns InvalidArgumentError if no TFLite is found in that section.
absl::Status ReadTFLiteFromSection(
    const std::string& litertlm_path, int section_idx,
    std::unique_ptr<tflite::FlatBufferModel>* tflite_model);

// Read any TF Lite from the file (convenience function if the caller knows
// that only 1 TF Lite file exists in the LiteRT-LM file). This function will
// not return an error if there are more than 1 TF Lite sections.
absl::Status ReadAnyTFLite(
    const std::string& litertlm_path,
    std::unique_ptr<tflite::FlatBufferModel>* tflite_model);

// Read a LlmMetadata from the specified section in the LiteRT-LM file.
// Returns InvalidArgumentError if no LlmMetadata are found in that section.
absl::Status ReadLlmMetadataFromSection(const std::string& litertlm_path,
                                        int section_idx,
                                        LlmMetadata* llm_metadata);

// Read any LlmMetadata from the file (convenience function if the caller
// knows that only 1 LlmMetadata proto exists in the LiteRT-LM file).
absl::Status ReadAnyLlmMetadata(const std::string& litertlm_path,
                                LlmMetadata* llm_metadata);

// Read a SP Tokenizer from the specified section in the LiteRT-LM file.
// Returns InvalidArgumentError if no SP Tokenizer is found in that section.
absl::Status ReadSPTokenizerFromSection(
    const std::string& litertlm_path, int section_idx,
    sentencepiece::SentencePieceProcessor* sp_proc);

// Read any SP Tokenizer from the file (convenience function if the caller knows
// that only 1 SP Tokenizer exists in the LiteRT-LM file).
absl::Status ReadAnySPTokenizer(const std::string& litertlm_path,
                                sentencepiece::SentencePieceProcessor* sp_proc);

}  // end namespace schema
}  // end namespace litertlm
}  // end namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LM_SCHEMA_CORE_LITERTLM_READ_H_
