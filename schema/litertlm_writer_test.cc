#include <cerrno>
#include <cstdint>
#include <cstring>
#include <filesystem>  // NOLINT: Required for path manipulation.
#include <fstream>
#include <ios>
#include <sstream>
#include <string>
#include <vector>

#include <gmock/gmock.h>  // For matchers like HasSubstr
#include <gtest/gtest.h>
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/proto/llm_metadata.pb.h"  // For LlmMetadata
#include "schema/core/litertlm_print.h"
#include "schema/litertlm_writer_utils.h"
#include "google/protobuf/text_format.h"  // from @com_google_protobuf  // For TextFormat::PrintToString

namespace litert {
namespace litertlm {
namespace schema {
namespace {

// Test fixture for LiteRTLMWrite tests, managing temporary resources.
class LiteRTLMWriteTest : public ::testing::Test {
 protected:
  std::string
      temp_dir_path_;  // Path to the unique temporary directory for the test.

  void SetUp() override {
    // Create a unique temporary directory using mkdtemp.
    // mkdtemp requires a C-style string template ending in XXXXXX.
    std::string temp_dir_template_str = "/tmp/litertlm_write_test_XXXXXX";

    // mkdtemp modifies the template string, so we need a writable buffer.
    std::vector<char> temp_dir_template_vec(temp_dir_template_str.begin(),
                                            temp_dir_template_str.end());
    temp_dir_template_vec.push_back('\0');  // Null-terminate
    char* temp_dir_c_str = temp_dir_template_vec.data();
    char* created_temp_dir_name = mkdtemp(temp_dir_c_str);

    if (created_temp_dir_name == nullptr) {
      FAIL() << "Failed to create temporary directory using mkdtemp. Template "
                "was: "
             << temp_dir_template_str << ". Error: " << strerror(errno);
    }
    temp_dir_path_ = created_temp_dir_name;
    ABSL_LOG(INFO) << "Temporary directory created: " << temp_dir_path_;
  }

  void TearDown() override {
    // Clean up the temporary directory and its contents using std::filesystem.
    ASSERT_TRUE(std::filesystem::exists(temp_dir_path_))
        << "Dir does not exist: " << temp_dir_path_;
    ASSERT_TRUE(std::filesystem::remove_all(temp_dir_path_))
        << "Unable to remove temp dir: " << temp_dir_path_;
  }

  void VerifyFile(const std::string& file_path) {
    // Check if the file exists.
    ASSERT_TRUE(std::filesystem::exists(file_path))
        << "File does not exist: " << file_path;

    // Get file size and check if it's greater than 0.
    uintmax_t fsize = std::filesystem::file_size(file_path);
    ASSERT_GT(fsize, 0) << "File " << file_path << " has size 0 or less.";
  }

  // Helper function to create a dummy file with specified content.
  void CreateDummyFile(const std::string& file_path,
                       const std::string& content) {
    std::ofstream outfile(file_path,
                          std::ios::binary);  // Open in binary for consistency
    ASSERT_TRUE(outfile.is_open())
        << "Failed to open file for writing: " << file_path;
    outfile << content;
    outfile.close();
    VerifyFile(file_path);
  }
};

// Test case: Successful creation with multiple files and metadata.
TEST_F(LiteRTLMWriteTest, BasicFileCreationAndValidation) {
  // 1. Define paths for temporary input files and the output file.
  const std::string tokenizer_path = temp_dir_path_ + "/tokenizer.spiece";
  const std::string tflite_model_path = temp_dir_path_ + "/model.tflite";
  const std::string llm_metadata_path = temp_dir_path_ + "/llm_metadata.pbtext";
  const std::string output_litertlm_path = temp_dir_path_ + "/output.litertlm";

  // 2. Create dummy input files.
  CreateDummyFile(tokenizer_path, "Dummy SentencePiece Model Content");
  CreateDummyFile(tflite_model_path,
                  "Dummy TFLite Model Content. Not a real model.");

  litert::lm::proto::LlmMetadata metadata;
  const std::string start_token = "<start>";
  const std::vector<std::string> stop_tokens = {"<stop>", "<eos>"};
  metadata.mutable_start_token()->set_token_str(start_token);

  // Set the stop_tokens
  for (const std::string& stop_token : stop_tokens) {
    metadata.add_stop_tokens()->set_token_str(stop_token);
  }

  std::string params_pbtext_content;
  ASSERT_TRUE(
      proto2::TextFormat::PrintToString(metadata, &params_pbtext_content));
  CreateDummyFile(llm_metadata_path, params_pbtext_content);

  // 3. Prepare arguments for LitertLmWrite.
  const std::vector<std::string> command_args = {
      tokenizer_path, tflite_model_path, llm_metadata_path};
  const std::string section_metadata_str =
      "tokenizer:tok_version=1.2,lang=en;"
      "tflite:model_size=2048,quantized=false;"
      "llm_metadata:author=TestyMcTestface,temperature=0.8";

  // 4. Call LitertLmWrite.
  const absl::Status result =
      LitertLmWrite(command_args, section_metadata_str, output_litertlm_path);
  ASSERT_TRUE(result.ok()) << "LitertLmWrite failed: " << result.message();

  // 5. Verify the output file is good.
  ASSERT_TRUE(std::filesystem::exists(output_litertlm_path))
      << "Output LiteRT-LM file was not created.";
  VerifyFile(output_litertlm_path);

  // 6. Inspect the content of the generated .litertlm file.
  std::stringstream inspection_output_ss;
  const absl::Status print_result =
      ProcessLiteRTLMFile(output_litertlm_path, inspection_output_ss);
  ASSERT_TRUE(print_result.ok())
      << "ProcessLiteRTLMFile failed: " << print_result.message();

  const std::string inspection_str = inspection_output_ss.str();
  ASSERT_FALSE(inspection_str.empty())
      << "ProcessLiteRTLMFile produced empty output.";

  // Check for presence of section types.
  EXPECT_THAT(inspection_str,
              testing::HasSubstr("AnySectionDataType_SP_Tokenizer"));
  EXPECT_THAT(inspection_str,
              testing::HasSubstr("AnySectionDataType_TFLiteModel"));
  EXPECT_THAT(inspection_str,
              testing::HasSubstr("AnySectionDataType_LlmMetadataProto"));

  // Check for presence of metadata (adjust based on ProcessLiteRTLMFile's
  // output format). Assuming ProcessLiteRTLMFile prints metadata like "key:
  // value".
  EXPECT_THAT(inspection_str, testing::HasSubstr("tok_version,"));
  EXPECT_THAT(inspection_str, testing::HasSubstr("(Float): 1.2"));
  EXPECT_THAT(inspection_str, testing::HasSubstr("lang,"));
  EXPECT_THAT(inspection_str, testing::HasSubstr("(String): en"));
  EXPECT_THAT(inspection_str, testing::HasSubstr("model_size,"));
  EXPECT_THAT(inspection_str, testing::HasSubstr("(Int32): 2048"));
  EXPECT_THAT(inspection_str, testing::HasSubstr("quantized,"));
  EXPECT_THAT(inspection_str, testing::HasSubstr("(Bool): 0"));
  EXPECT_THAT(inspection_str, testing::HasSubstr("author,"));
  EXPECT_THAT(inspection_str, testing::HasSubstr("(String): TestyMcTestface"));
  EXPECT_THAT(inspection_str, testing::HasSubstr("temperature,"));
  EXPECT_THAT(inspection_str, testing::HasSubstr("(Float): 0.8"));
}

// Test case: No section metadata provided.
TEST_F(LiteRTLMWriteTest, NoMetadataTest) {
  const std::string tokenizer_path = temp_dir_path_ + "/tokenizer.spiece";
  const std::string output_litertlm_path =
      temp_dir_path_ + "/output_no_meta.litertlm";

  CreateDummyFile(tokenizer_path, "Some tokenizer data.");

  const std::vector<std::string> command_args = {tokenizer_path};
  const std::string section_metadata_str = "";  // Empty metadata string.

  const absl::Status result =
      LitertLmWrite(command_args, section_metadata_str, output_litertlm_path);
  ASSERT_TRUE(result.ok()) << "LitertLmWrite failed with no metadata: "
                           << result.message();
  VerifyFile(output_litertlm_path);

  std::stringstream inspection_output_ss;
  const absl::Status print_result =
      ProcessLiteRTLMFile(output_litertlm_path, inspection_output_ss);
  ASSERT_TRUE(print_result.ok())
      << "ProcessLiteRTLMFile failed: " << print_result.message();
  EXPECT_THAT(inspection_output_ss.str(),
              testing::HasSubstr("AnySectionDataType_SP_Tokenizer"));
  // Ensure no accidental metadata from other tests is present.
  EXPECT_THAT(inspection_output_ss.str(),
              testing::Not(testing::HasSubstr("tok_version")));
}

// Test case: Mismatched order between input files and section_metadata.
TEST_F(LiteRTLMWriteTest, MismatchedMetadataOrderTest) {
  const std::string tokenizer_path = temp_dir_path_ + "/tokenizer.spiece";
  const std::string tflite_model_path = temp_dir_path_ + "/model.tflite";
  const std::string output_litertlm_path =
      temp_dir_path_ + "/output_mismatch_order.litertlm";

  CreateDummyFile(tokenizer_path, "Tokenizer data");
  CreateDummyFile(tflite_model_path, "TFLite data");

  const std::vector<std::string> command_args = {
      tokenizer_path,    // File 1: tokenizer
      tflite_model_path  // File 2: tflite
  };
  // Metadata order is tflite (for file 1) then tokenizer (for file 2), which is
  // a mismatch.
  const std::string section_metadata_str =
      "tflite:key1=val1;"
      "tokenizer:key2=val2";

  const absl::Status result =
      LitertLmWrite(command_args, section_metadata_str, output_litertlm_path);
  ASSERT_FALSE(result.ok())
      << "LitertLmWrite should have failed due to mismatched metadata order.";
  EXPECT_THAT(result.message(),
              testing::HasSubstr("Order mismatch for section at index 0"));
  EXPECT_THAT(result.message(),
              testing::HasSubstr("Expected section from filename: 'tokenizer', "
                                 "Found in metadata: 'tflite'"));
  ASSERT_FALSE(std::filesystem::exists(output_litertlm_path))
      << "Output file should not be created on failure.";
}

// Test case: Unsupported file extension.
TEST_F(LiteRTLMWriteTest, InvalidFileExtensionTest) {
  const std::string invalid_file_path = temp_dir_path_ + "/document.txt";
  const std::string output_litertlm_path =
      temp_dir_path_ + "/output_invalid_ext.litertlm";
  CreateDummyFile(invalid_file_path, "This is just a text file.");

  const std::vector<std::string> command_args = {invalid_file_path};
  const std::string section_metadata_str =
      "";  // No metadata needed for this failure.

  const absl::Status result =
      LitertLmWrite(command_args, section_metadata_str, output_litertlm_path);
  ASSERT_FALSE(result.ok());
  EXPECT_THAT(result.message(),
              testing::HasSubstr("Unsupported file extension for:"));
  EXPECT_THAT(result.message(), testing::HasSubstr("document.txt"));
}

// Test case: No input files provided.
TEST_F(LiteRTLMWriteTest, EmptyCommandArgsTest) {
  const std::string output_litertlm_path =
      temp_dir_path_ + "/output_no_input_files.litertlm";
  const std::vector<std::string> command_args = {};  // Empty list of files.
  const std::string section_metadata_str = "";

  const absl::Status result =
      LitertLmWrite(command_args, section_metadata_str, output_litertlm_path);
  ASSERT_FALSE(result.ok());
  EXPECT_THAT(result.message(),
              testing::HasSubstr("At least one input file must be provided."));
}

// Test case: Input file path does not exist.
TEST_F(LiteRTLMWriteTest, NonExistentInputFileTest) {
  const std::string non_existent_file_path =
      temp_dir_path_ + "/i_do_not_exist.spiece";
  const std::string output_litertlm_path =
      temp_dir_path_ + "/output_non_existent_input.litertlm";

  const std::vector<std::string> command_args = {non_existent_file_path};
  // Metadata must match the number of command_args if provided, even if file
  // doesn't exist.
  const std::string section_metadata_str = "tokenizer:key=val";

  const absl::Status result =
      LitertLmWrite(command_args, section_metadata_str, output_litertlm_path);
  ASSERT_FALSE(result.ok());
  // Error message depends on which file type fails first during parsing.
  // For .spiece (FileBackedSectionStream): "Failed to open file"
  // For .pbtext (std::ifstream): "Could not open llm_metadata text file"
  EXPECT_THAT(
      std::string(result.message()),
      testing::AnyOf(
          testing::HasSubstr("Could not open"),  // Covers .pb, .proto, .pbtext
          testing::HasSubstr(
              "Failed to open file")  // Covers .spiece, .tflite via
                                      // FileBackedSectionStream
          ));
}

// Test case: Invalid format for section_metadata (missing colon).
TEST_F(LiteRTLMWriteTest, InvalidMetadataFormatMissingColonTest) {
  const std::string tokenizer_path = temp_dir_path_ + "/tokenizer.spiece";
  const std::string output_litertlm_path =
      temp_dir_path_ + "/output_bad_meta_format1.litertlm";
  CreateDummyFile(tokenizer_path, "Tokenizer data");

  const std::vector<std::string> command_args = {tokenizer_path};
  const std::string section_metadata_str =
      "tokenizer_no_colon_key=value";  // Invalid format.

  const absl::Status result =
      LitertLmWrite(command_args, section_metadata_str, output_litertlm_path);
  ASSERT_FALSE(result.ok());
  EXPECT_THAT(result.message(),
              testing::HasSubstr("Invalid section metadata format"));
}

// Test case: Invalid format for section_metadata (bad key-value pair).
TEST_F(LiteRTLMWriteTest, InvalidMetadataFormatBadKeyValuePairTest) {
  const std::string tokenizer_path = temp_dir_path_ + "/tokenizer.spiece";
  const std::string output_litertlm_path =
      temp_dir_path_ + "/output_bad_meta_format2.litertlm";
  CreateDummyFile(tokenizer_path, "Tokenizer data");

  const std::vector<std::string> command_args = {tokenizer_path};
  const std::string section_metadata_str =
      "tokenizer:key_without_equals_sign";  // Invalid key-value.

  const absl::Status result =
      LitertLmWrite(command_args, section_metadata_str, output_litertlm_path);
  ASSERT_FALSE(result.ok());
  EXPECT_THAT(result.message(),
              testing::HasSubstr("Failed to parse key-value pair"));
  EXPECT_THAT(result.message(), testing::HasSubstr("key_without_equals_sign"));
}

// Test case: Number of metadata sections doesn't match number of input files.
TEST_F(LiteRTLMWriteTest, MetadataCountMismatchTest) {
  const std::string tokenizer_path = temp_dir_path_ + "/tokenizer.spiece";
  const std::string output_litertlm_path =
      temp_dir_path_ + "/output_meta_count_mismatch.litertlm";
  CreateDummyFile(tokenizer_path, "Tokenizer data");

  const std::vector<std::string> command_args = {
      tokenizer_path};  // 1 input file.
  const std::string section_metadata_str =
      "tokenizer:k1=v1;tflite:k2=v2";  // 2 metadata sections.

  const absl::Status result =
      LitertLmWrite(command_args, section_metadata_str, output_litertlm_path);
  ASSERT_FALSE(result.ok());
  EXPECT_THAT(result.message(),
              testing::HasSubstr("Mismatch in number of sections"));
  EXPECT_THAT(result.message(), testing::HasSubstr("input files (1)"));
  EXPECT_THAT(result.message(), testing::HasSubstr("section_metadata (2)"));
}

}  // namespace
}  // namespace schema
}  // namespace litertlm
}  // namespace litert
