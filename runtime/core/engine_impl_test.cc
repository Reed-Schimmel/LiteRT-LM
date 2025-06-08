// Copyright 2025 The ODML Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cstdlib>
#include <filesystem>  // NOLINT: Required for path manipulation.
#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/proto/sampler_params.pb.h"
#include "runtime/util/test_utils.h"  // NOLINT

namespace litert::lm {
namespace {

TEST(EngineTest, CreateEngine_WithoutCache) {
  auto task_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm_new_metadata.task";
  auto model_assets = ModelAssets::Create(task_path.string());
  ASSERT_OK(model_assets);
  auto engine_settings =
      EngineSettings::CreateDefault(*model_assets, Backend::CPU);
  ASSERT_OK(engine_settings);
  engine_settings->GetMutableMainExecutorSettings().SetMaxNumTokens(160);
  engine_settings->GetMutableMainExecutorSettings().SetCacheDir(":nocache");

  absl::StatusOr<std::unique_ptr<Engine>> llm =
      Engine::CreateEngine(*engine_settings);
  ABSL_CHECK_OK(llm);

  absl::StatusOr<std::unique_ptr<Engine::Session>> session =
      (*llm)->CreateSession(SessionConfig::CreateDefault());
  ABSL_CHECK_OK(session);

  ABSL_CHECK_OK((*session)->RunPrefill({InputText("Hello world!")}));

  auto responses = (*session)->RunDecode();
  EXPECT_OK(responses);
  EXPECT_EQ(responses->GetNumOutputCandidates(), 1);
  EXPECT_FALSE(responses->GetResponseTextAt(0)->empty());
}

TEST(EngineTest, CreateEngine_WithCache) {
  auto cache_path = std::filesystem::path(::testing::TempDir()) /
       absl::StrCat("cache-", std::rand());
  std::filesystem::remove_all(cache_path);
  absl::Cleanup remove_cache = [cache_path] {
    std::filesystem::remove_all(cache_path);
  };

  auto task_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm_new_metadata.task";
  auto model_assets = ModelAssets::Create(task_path.string());
  ASSERT_OK(model_assets);
  auto engine_settings =
      EngineSettings::CreateDefault(*model_assets, Backend::CPU);
  ASSERT_OK(engine_settings);
  engine_settings->GetMutableMainExecutorSettings().SetMaxNumTokens(160);
  engine_settings->GetMutableMainExecutorSettings().SetCacheDir(
      cache_path.string());

  // 1st run to populate the cache.
  absl::StatusOr<std::unique_ptr<Engine>> llm =
      Engine::CreateEngine(*engine_settings);
  ABSL_CHECK_OK(llm);

  absl::StatusOr<std::unique_ptr<Engine::Session>> session =
      (*llm)->CreateSession(SessionConfig::CreateDefault());
  ABSL_CHECK_OK(session);

  ABSL_CHECK_OK((*session)->RunPrefill({InputText("Hello world!")}));

  auto responses = (*session)->RunDecode();
  EXPECT_OK(responses);
  EXPECT_EQ(responses->GetNumOutputCandidates(), 1);
  EXPECT_FALSE(responses->GetResponseTextAt(0)->empty());

#if 0  // Disabled due to ASAN failure. Further investigation is needed.
  // 2nd run with the same engine and the same cache.
  session->reset();
  session = (*llm)->CreateSession(SessionConfig::CreateDefault());
  ABSL_CHECK_OK(session);

  ABSL_CHECK_OK((*session)->RunPrefill({InputText("Hello world!")}));

  responses = (*session)->RunDecode();
  EXPECT_OK(responses);
  EXPECT_EQ(responses->GetNumOutputCandidates(), 1);
  EXPECT_FALSE(responses->GetResponseTextAt(0)->empty());

  // 3rd run with a new engine and the same cache.
  session->reset();
  llm->reset();
  llm = Engine::CreateEngine(*engine_settings);
  ABSL_CHECK_OK(llm);

  session = (*llm)->CreateSession(SessionConfig::CreateDefault());
  ABSL_CHECK_OK(session);

  ABSL_CHECK_OK((*session)->RunPrefill({InputText("Hello world!")}));

  responses = (*session)->RunDecode();
  EXPECT_OK(responses);
  EXPECT_EQ(responses->GetNumOutputCandidates(), 1);
  EXPECT_FALSE(responses->GetResponseTextAt(0)->empty());
#endif
}

// TODO (b/397975034): Add more tests for Engine.

}  // namespace
}  // namespace litert::lm
