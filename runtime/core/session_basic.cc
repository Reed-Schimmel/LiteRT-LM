#include "runtime/core/session_basic.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "runtime/components/sampler.h"
#include "runtime/components/sampler_factory.h"
#include "runtime/components/stop_token_detector.h"
#include "runtime/components/tokenizer.h"
#include "runtime/core/pipeline.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/llm_executor.h"
#include "runtime/framework/threadpool.h"
#include "runtime/proto/sampler_params.pb.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/status_macros.h"  // IWYU pragma: keep

namespace litert::lm {

// static
absl::StatusOr<std::unique_ptr<SessionBasic>> SessionBasic::Create(
    LlmExecutor* executor, Tokenizer* tokenizer,
    const SessionConfig& session_config,
    std::optional<BenchmarkInfo> benchmark_info,
    ThreadPool* worker_thread_pool) {
  auto sampler_backend = session_config.GetSamplerBackend();
  std::unique_ptr<Sampler> sampler;
  // If use CPU sampling, we create it here; For GPU sampling, we let executor
  // create it internally.
  if (sampler_backend == Backend::CPU) {
    ASSIGN_OR_RETURN(
        sampler,
        CreateSampler(sampler_backend, session_config.GetNumOutputCandidates(),
                      session_config.GetSamplerParams()));
  } else if (sampler_backend != Backend::GPU &&
             sampler_backend != Backend::NPU) {
    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported sampler backend: ", sampler_backend));
  }

  if (benchmark_info.has_value()) {
    ABSL_LOG(INFO) << "Benchmark is enabled.";
  }
  StopTokenDetector stop_token_detector(
      session_config.GetNumOutputCandidates());
  for (const auto& stop_token_sequence : session_config.GetStopTokenIds()) {
    RETURN_IF_ERROR(
        stop_token_detector.AddStopTokenSequence(stop_token_sequence));
  }
  return absl::WrapUnique(new SessionBasic(
      executor, tokenizer, std::move(sampler), session_config, benchmark_info,
      worker_thread_pool, stop_token_detector));
}

SessionBasic::~SessionBasic() {
  auto status = executor_.Reset();
  if (!status.ok()) {
    ABSL_LOG(ERROR) << "Failed to reset executor: " << status;
  }
}

absl::Status SessionBasic::PrefillInternal(absl::string_view input,
                                           bool wait_for_completion) {
  // TODO(b/397975034): Consider to utilize a prompt formatting logic in a
  // separate library/class.
  // Update the input with prompt formatting.
  std::string formatted_input =
      absl::StrCat(session_config_.GetPromptTemplates().user().prefix(), input,
                   session_config_.GetPromptTemplates().user().suffix(),
                   session_config_.GetPromptTemplates().model().prefix());
  ABSL_LOG(INFO) << "PrefillInternal: " << formatted_input;
  ASSIGN_OR_RETURN(last_prefill_token_id_,
                   Prefill(executor_, tokenizer_, formatted_input,
                           session_config_.GetStartTokenId(),
                           wait_for_completion, benchmark_info_));
  return absl::OkStatus();
}

absl::Status SessionBasic::RunPrefill(const std::vector<InputData>& contents) {
  if (contents.empty()) {
    return absl::InvalidArgumentError("Input is empty.");
  }
  absl::Status status;
  for (const auto& input : contents) {
    RETURN_IF_ERROR(worker_thread_pool_.Schedule(
        [this, input_copy = ToString(input).value(), &status]() {
          status =
              this->PrefillInternal(input_copy, /*wait_for_completion=*/true);
        }));
  }
  RETURN_IF_ERROR(worker_thread_pool_.WaitUntilDone(Engine::kDefaultTimeout));
  return status;
}

absl::Status SessionBasic::RunPrefillAsync(
    const std::vector<InputData>& contents, InferenceObservable* observer) {
  if (contents.empty()) {
    return absl::InvalidArgumentError("Input is empty.");
  }
  for (const auto& input : contents) {
    RETURN_IF_ERROR(worker_thread_pool_.Schedule(
        [this, input_copy = ToString(input).value(), observer]() {
          absl::Status status =
              this->PrefillInternal(input_copy, /*wait_for_completion=*/false);
          ABSL_LOG(INFO) << "RunPrefillAsync status: " << status;
          if (status.ok()) {
            observer->OnDone();
          } else {
            observer->OnError(status);
          }
        }));
  }
  return absl::OkStatus();
}

absl::StatusOr<Responses> SessionBasic::DecodeInternal() {
  if (sampler_ == nullptr) {
    ASSIGN_OR_RETURN(
        auto responses,
        Decode(executor_, tokenizer_, stop_token_detector_, benchmark_info_));
    return responses;
  } else {
    std::vector<int> decoded_ids(session_config_.GetNumOutputCandidates(),
                                 last_prefill_token_id_);
    auto decoded_ids_buffer = CopyToTensorBuffer<int>(
        decoded_ids, {session_config_.GetNumOutputCandidates(), 1});
    ASSIGN_OR_RETURN(
        auto responses,
        DecodeCustomSampling(executor_, tokenizer_, stop_token_detector_,
                             /*num_output_candidates=*/1, *sampler_,
                             *decoded_ids_buffer, benchmark_info_));
    return responses;
  }
}

absl::Status SessionBasic::DecodeInternalStreaming(
    InferenceObservable* observer) {
  if (sampler_ == nullptr) {
    RETURN_IF_ERROR(DecodeStreaming(executor_, tokenizer_, stop_token_detector_,
                                    benchmark_info_, observer));
  } else {
    std::vector<int> decoded_ids(session_config_.GetNumOutputCandidates(),
                                 last_prefill_token_id_);
    auto decoded_ids_buffer = CopyToTensorBuffer<int>(
        decoded_ids, {session_config_.GetNumOutputCandidates(), 1});
    RETURN_IF_ERROR(DecodeCustomSamplingStreaming(
        executor_, tokenizer_, stop_token_detector_,
        /*num_output_candidates=*/1, *sampler_, *decoded_ids_buffer,
        benchmark_info_, observer));
  }
  return absl::OkStatus();
}

absl::StatusOr<Responses> SessionBasic::RunDecode() {
  ABSL_LOG(INFO) << "RunDecodeSync";
  absl::StatusOr<Responses> responses;
  RETURN_IF_ERROR(worker_thread_pool_.Schedule(
      [this, &responses]() { responses = this->DecodeInternal(); }));
  RETURN_IF_ERROR(worker_thread_pool_.WaitUntilDone(Engine::kDefaultTimeout));
  return responses;
}

absl::Status SessionBasic::RunDecodeAsync(InferenceObservable* observer) {
  ABSL_LOG(INFO) << "RunDecodeAsync";
  return worker_thread_pool_.Schedule([this, observer]() {
    this->DecodeInternalStreaming(observer).IgnoreError();
  });
}

absl::StatusOr<Responses> SessionBasic::GenerateContent(
    const std::vector<InputData>& contents) {
  RETURN_IF_ERROR(RunPrefill(contents));
  return RunDecode();
}

absl::Status SessionBasic::GenerateContentStream(
    const std::vector<InputData>& contents, InferenceObservable* observer) {
  RETURN_IF_ERROR(RunPrefillAsync(contents, observer));
  return RunDecodeAsync(observer);
}

absl::StatusOr<BenchmarkInfo> SessionBasic::GetBenchmarkInfo() {
  if (benchmark_info_.has_value()) {
    return benchmark_info_.value();
  }
  return absl::InternalError(
      "Benchmark is not enabled. Please make sure the BenchmarkParams is set "
      "in the EngineSettings.");
}

}  // namespace litert::lm
