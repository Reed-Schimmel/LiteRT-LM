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

#include "runtime/executor/llm_litert_compiled_model_executor.h"

#include <cstdlib>
#include <filesystem>  // NOLINT: Required for path manipulation.
#include <memory>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/components/model_resources.h"
#include "runtime/components/model_resources_task.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/util/model_asset_bundle_resources.h"
#include "runtime/util/scoped_file.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep

namespace litert::lm {
namespace {
const int kMaxNumTokens = 32;
const int kNumThreads = 4;

using ::litert::lm::Backend;
using ::litert::lm::LlmExecutorSettings;
using ::litert::lm::ModelAssets;
using ::litert::lm::LlmLiteRtCompiledModelExecutor;
using ::litert::lm::ModelAssetBundleResources;
using ::litert::lm::ModelResourcesTask;

absl::StatusOr<std::unique_ptr<ModelResources>> CreateExecutorModelResources(
    absl::string_view model_path) {
  auto scoped_file = ScopedFile::Open(model_path);
  auto resources = ModelAssetBundleResources::Create(
      /*tag=*/"", std::move(*scoped_file));
  auto model_resources = ModelResourcesTask::Create(std::move(*resources));
  return model_resources;
}

TEST(LlmLiteRTCompiledModelExecutorTest, CreateExecutorTest_WithoutCache) {
  auto model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm.task";
  ASSERT_OK_AND_ASSIGN(auto model_resources,
                       CreateExecutorModelResources(model_path.string()));
  auto model_assets = ModelAssets::Create(model_path.string());
  ASSERT_OK(model_assets);
  auto executor_settings =
      LlmExecutorSettings::CreateDefault(*model_assets, Backend::CPU);
  executor_settings->SetCacheDir(":nocache");
  executor_settings->SetMaxNumTokens(kMaxNumTokens);
  ::litert::lm::CpuConfig config;
  config.number_of_threads = kNumThreads;
  executor_settings->SetBackendConfig(config);
  auto executor = LlmLiteRtCompiledModelExecutor::Create(
      *executor_settings, std::move(model_resources));
  ASSERT_OK(executor);
  ASSERT_NE(*executor, nullptr);
}

TEST(LlmLiteRTCompiledModelExecutorTest, CreateExecutorTest_WithCache) {
  auto cache_path = std::filesystem::path(::testing::TempDir()) /
       absl::StrCat("cache-", std::rand());
  std::filesystem::remove_all(cache_path);
  absl::Cleanup remove_cache = [cache_path] {
    std::filesystem::remove_all(cache_path);
  };

  auto model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm.task";
  ASSERT_OK_AND_ASSIGN(auto model_resources,
                       CreateExecutorModelResources(model_path.string()));
  auto model_assets = ModelAssets::Create(model_path.string());
  ASSERT_OK(model_assets);
  auto executor_settings =
      LlmExecutorSettings::CreateDefault(*model_assets, Backend::CPU);
  executor_settings->SetCacheDir(cache_path.string());
  executor_settings->SetMaxNumTokens(kMaxNumTokens);
  ::litert::lm::CpuConfig config;
  config.number_of_threads = kNumThreads;
  executor_settings->SetBackendConfig(config);
  auto executor = LlmLiteRtCompiledModelExecutor::Create(
      *executor_settings, std::move(model_resources));
  ASSERT_OK(executor);
  ASSERT_NE(*executor, nullptr);
}

}  // namespace
}  // namespace litert::lm
