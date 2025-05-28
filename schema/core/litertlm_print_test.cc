#include "schema/core/litertlm_print.h"

#include <filesystem>  // NOLINT: Required for path manipulation.
#include <sstream>
#include <string>

#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl

namespace litert {
namespace litertlm {
namespace schema {
namespace {

TEST(LiteRTLMPrintTest, ProcessLiteRTLMFileTest) {
  const std::string input_filename =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/schema/testdata/test_tok_tfl_llm.litertlm";

  std::stringstream output_ss;
  absl::Status result = ProcessLiteRTLMFile(input_filename, output_ss);
  ASSERT_TRUE(result.ok());
  ASSERT_GT(output_ss.str().size(), 0);
  ASSERT_NE(output_ss.str().find("AnySectionDataType_TFLiteModel"),
            std::string::npos);
  ASSERT_NE(output_ss.str().find("AnySectionDataType_SP_Tokenizer"),
            std::string::npos);
  ASSERT_NE(output_ss.str().find("AnySectionDataType_LlmMetadataProto"),
            std::string::npos);
}

}  // namespace
}  // namespace schema
}  // namespace litertlm
}  // namespace litert
