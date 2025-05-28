#include "schema/core/litertlm_read.h"

#include <filesystem>  // NOLINT: Required for path manipulation.
#include <fstream>
#include <ios>
#include <memory>
#include <string>

#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "runtime/proto/llm_metadata.pb.h"
#include "schema/core/litertlm_header_schema_generated.h"
#include "sentencepiece_processor.h"  // from @sentencepiece
#include "tensorflow/lite/model_builder.h"  // from @org_tensorflow

namespace litert {
namespace litertlm {
namespace schema {
namespace {

TEST(LiteRTLMReadTest, HeaderReadFile) {
  const std::string input_filename =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/schema/testdata/test_tok_tfl_llm.litertlm";

  LitertlmHeader header;
  int major_version, minor_version, patch_version;

  absl::Status status = ReadHeaderFromLiteRTLM(
      input_filename, &header, &major_version, &minor_version, &patch_version);

  ASSERT_TRUE(status.ok());
  const LiteRTLMMetaData* metadata = header.metadata;
  auto system_metadata = metadata->system_metadata();
  ASSERT_TRUE(!!system_metadata);
  auto entries = system_metadata->entries();
  ASSERT_TRUE(!!entries);         // Ensure entries is not null
  ASSERT_EQ(entries->size(), 2);  // Check the number of key-value pairs.
}

TEST(LiteRTLMReadTest, HeaderReadIstream) {
  const std::string input_filename =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/schema/testdata/test_tok_tfl_llm.litertlm";

  LitertlmHeader header;
  int major_version, minor_version, patch_version;

  std::ifstream input_file_stream(input_filename, std::ios::binary);
  ASSERT_TRUE(input_file_stream.is_open());
  absl::Status status =
      ReadHeaderFromLiteRTLM(input_file_stream, &header, &major_version,
                             &minor_version, &patch_version);
  ASSERT_TRUE(status.ok());
  const LiteRTLMMetaData* metadata = header.metadata;
  auto system_metadata = metadata->system_metadata();
  ASSERT_TRUE(!!system_metadata);
  auto entries = system_metadata->entries();
  ASSERT_TRUE(!!entries);         // Ensure entries is not null
  ASSERT_EQ(entries->size(), 2);  // Check the number of key-value pairs.
}

TEST(LiteRTLMReadTest, TokenizerRead) {
  const std::string input_filename =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/schema/testdata/test_tok_tfl_llm.litertlm";

  sentencepiece::SentencePieceProcessor sp_proc;
  absl::Status result = ReadSPTokenizerFromSection(input_filename, 0, &sp_proc);
  ASSERT_TRUE(result.ok());
}

TEST(LiteRTLMReadTest, LlmMetadataRead) {
  using litert::lm::proto::LlmMetadata;
  const std::string input_filename =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/schema/testdata/test_tok_tfl_llm.litertlm";

  LlmMetadata params;
  absl::Status result = ReadLlmMetadataFromSection(input_filename, 2, &params);
  ASSERT_TRUE(result.ok());
}

TEST(LiteRTLMReadTest, TFLiteRead) {
  const std::string input_filename =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/schema/testdata/test_tok_tfl_llm.litertlm";

  std::unique_ptr<tflite::FlatBufferModel> model;
  absl::Status result = ReadTFLiteFromSection(input_filename, 1, &model);
  ASSERT_TRUE(result.ok());
}

TEST(LiteRTLMReadTest, TFLiteReadAny) {
  const std::string input_filename =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/schema/testdata/test_tok_tfl_llm.litertlm";

  std::unique_ptr<tflite::FlatBufferModel> tflite_model;
  absl::Status result = ReadAnyTFLite(input_filename, &tflite_model);
  ASSERT_TRUE(result.ok());
}

TEST(LiteRTLMReadTest, TFLiteRead_InvalidSection) {
  const std::string input_filename =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/schema/testdata/test_tok_tfl_llm.litertlm";

  std::unique_ptr<tflite::FlatBufferModel> tflite_model;
  absl::Status result = ReadTFLiteFromSection(input_filename, 0, &tflite_model);
  ASSERT_FALSE(result.ok());
  ASSERT_EQ(result.code(), absl::StatusCode::kInvalidArgument);
}

}  // namespace
}  // namespace schema
}  // namespace litertlm
}  // namespace litert
