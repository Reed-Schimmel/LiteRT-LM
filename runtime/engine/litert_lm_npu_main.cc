// A simple command line tool to run the litert LLM engine on NPU.

#include <cstdint>
#include <filesystem>  // NOLINT: Required for path manipulation.
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/flags/parse.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "third_party/odml/infra/genai/inference/executor/llm_litert_npu_compiled_model_executor.h"
#include "runtime/components/sentencepiece_tokenizer.h"
#include "runtime/components/tokenizer.h"
#include "runtime/core/session_basic.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/proto/sampler_params.pb.h"
#include "tensorflow/lite/profiling/time.h"  // from @org_tensorflow

ABSL_FLAG(std::string, gemma3_path, "", "Path to the Gemma3 model.");
ABSL_FLAG(std::string, embedder_path, "", "Path to the embedder model.");
ABSL_FLAG(std::string, auxiliary_path, "", "Path to the auxiliary model.");
ABSL_FLAG(std::string, tokenizer_path, "", "Path to the tokenizer model.");
ABSL_FLAG(std::string, litert_dispatch_lib_path, "",
          "Path to the LiteRT dispatch library.");
ABSL_FLAG(std::string, prompt, "", "Prompt to run.");
ABSL_FLAG(bool, gemma_only_quantized, true,
          "If only Gemma3 is quantized. If false, all models except embedder "
          "are quantized (*except the embedder for now).");

using odml::infra::LlmLiteRtNpuCompiledModelExecutor::ModelQuantization::
    kAllQuantized;
using odml::infra::LlmLiteRtNpuCompiledModelExecutor::ModelQuantization::
    kGemmaOnlyQuantized;

odml::infra::LlmLiteRtNpuCompiledModelExecutor::ModelQuantization
GetQuantizationSchema() {
  if (absl::GetFlag(FLAGS_gemma_only_quantized)) {
    return kGemmaOnlyQuantized;
  } else {
    return kAllQuantized;
  }
}

int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);

  // Create the tokenizer.
  auto tokenizer_or = litert::lm::SentencePieceTokenizer::CreateFromFile(
      absl::GetFlag(FLAGS_tokenizer_path));
  if (tokenizer_or.ok()) {
    ABSL_LOG(INFO) << "tokenizer created successfully";
  } else {
    ABSL_LOG(ERROR) << "tokenizer creation failed: " << tokenizer_or.status();
  }
  std::shared_ptr<litert::lm::Tokenizer> tokenizer =
      std::move(tokenizer_or.value());

  // Create the executor.
  uint64_t start = tflite::profiling::time::NowMicros();
  ABSL_LOG(INFO) << "Creating executor";
  auto executor_or = odml::infra::LlmLiteRtNpuCompiledModelExecutor::Create(
      GetQuantizationSchema(), absl::GetFlag(FLAGS_gemma3_path),
      absl::GetFlag(FLAGS_embedder_path), absl::GetFlag(FLAGS_auxiliary_path),
      absl::GetFlag(FLAGS_litert_dispatch_lib_path));
  uint64_t end = tflite::profiling::time::NowMicros();
  ABSL_LOG(INFO) << "executor creation took " << (end - start) << " us";
  if (executor_or.ok()) {
    ABSL_LOG(INFO) << "executor created successfully";
  } else {
    ABSL_LOG(ERROR) << "executor creation failed: " << executor_or.status();
  }
  std::unique_ptr<odml::infra::LlmLiteRtNpuCompiledModelExecutor> executor =
      std::move(executor_or.value());
  std::shared_ptr<odml::infra::LlmLiteRtNpuCompiledModelExecutor>
      executor_shared = std::move(executor);

  // Create the session.
  constexpr int kEndOfTurnTokenId = 106;
  std::vector<int> stop_token_ids = {kEndOfTurnTokenId};
  // TODO(b/405424188): The NPU executor currently uses direct sampling on the
  // int16 logits. Use SamplerParameters::TYPE_UNSPECIFIED to avoid using the
  // default float CPU sampler.
  // See the description of cl/752681117 that justifies the usage of a
  // custom sampler (decode performance speed-up). We should extend the
  // 'litert::lm' Sampler to support this natively and remove the custom sampler
  // of the NPU executor.
  // TODO(b/415915773): update the session config to use the provided sampler
  // once it supports the int16 logits.
  auto session_config = litert::lm::SessionConfig::CreateDefault();
  session_config.GetMutableSamplerParams().set_type(
      litert::lm::proto::SamplerParameters::TYPE_UNSPECIFIED);
  auto session = litert::lm::SessionBasic::Create(
      executor_shared, tokenizer, stop_token_ids, session_config, std::nullopt);

  // Run the session.
  const std::string prompt = absl::GetFlag(FLAGS_prompt);
  ABSL_LOG(INFO) << "Prompt: " << prompt;
  start = tflite::profiling::time::NowMicros();
  auto status = (*session)->RunPrefill(prompt);
  end = tflite::profiling::time::NowMicros();
  ABSL_LOG(INFO) << "RunPrefill took " << (end - start) << " us";

  start = tflite::profiling::time::NowMicros();
  auto responses = (*session)->RunDecode();
  end = tflite::profiling::time::NowMicros();

  ABSL_LOG(INFO) << "RunDecode took " << (end - start) << " us";
  if (responses.ok()) {
    for (int i = 0; i < responses->GetNumOutputCandidates(); ++i) {
      auto response_text = responses->GetResponseTextAt(i);
      ABSL_LOG(INFO) << "Generated response: " << (*response_text);
    }
  } else {
    ABSL_LOG(ERROR) << "response failed: " << responses.status();
  }

  executor_shared->PrintLatencyStats();

  return 0;
}
