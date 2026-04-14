# Copyright 2026 The ODML Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the LiteRT-LM serve command."""

import sys
from unittest import mock

from absl.testing import absltest

# Mock litert_lm before importing serve to avoid loading heavy C extensions.
mock_litert_lm = mock.MagicMock()
mock_litert_lm.Backend.CPU = "cpu"
sys.modules["litert_lm"] = (
    mock_litert_lm
)

# Also mock model as it imports litert_lm too.
mock_model_mod = mock.MagicMock()
sys.modules["litert_lm_cli.model"] = (
    mock_model_mod
)

from litert_lm_cli import serve


class ServeTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    # Reset global state in serve.py
    serve._current_engine = None
    serve._current_model_id = None
    # Reset mocks
    mock_litert_lm.Engine.reset_mock()
    mock_model_mod.Model.from_model_id.reset_mock()
    mock_model_mod.Model.from_model_id.side_effect = None

  def test_gemini_to_litertlm_message_user(self):
    gemini_content = {"role": "user", "parts": [{"text": "Hello"}]}
    expected = {"role": "user", "content": [{"type": "text", "text": "Hello"}]}
    self.assertEqual(serve.gemini_to_litertlm_message(gemini_content), expected)

  def test_gemini_to_litertlm_message_model(self):
    gemini_content = {"role": "model", "parts": [{"text": "Hi"}]}
    expected = {
        "role": "assistant",
        "content": [{"type": "text", "text": "Hi"}],
    }
    self.assertEqual(serve.gemini_to_litertlm_message(gemini_content), expected)

  def test_gemini_to_litertlm_message_default_role(self):
    gemini_content = {"parts": [{"text": "No role"}]}
    expected = {
        "role": "user",
        "content": [{"type": "text", "text": "No role"}],
    }
    self.assertEqual(serve.gemini_to_litertlm_message(gemini_content), expected)

  def test_litertlm_to_gemini_response(self):
    litertlm_response = {
        "role": "assistant",
        "content": [{"type": "text", "text": "Response text"}],
    }
    expected = {
        "candidates": [{
            "content": {
                "role": "model",
                "parts": [{"text": "Response text"}],
            },
            "finishReason": "STOP",
            "index": 0,
        }]
    }
    self.assertEqual(
        serve.litertlm_to_gemini_response(litertlm_response), expected
    )

  def test_litertlm_to_gemini_response_streaming(self):
    litertlm_response = {"content": [{"type": "text", "text": "Chunk"}]}
    expected = {
        "candidates": [{
            "content": {
                "role": "model",
                "parts": [{"text": "Chunk"}],
            },
            "index": 0,
        }]
    }
    self.assertEqual(
        serve.litertlm_to_gemini_response(litertlm_response, finish_reason=""),
        expected,
    )

  def test_litertlm_to_gemini_response_custom_finish_reason(self):
    litertlm_response = {"content": []}
    expected = {
        "candidates": [{
            "content": {
                "role": "model",
                "parts": [],
            },
            "finishReason": "MAX_TOKENS",
            "index": 0,
        }]
    }
    self.assertEqual(
        serve.litertlm_to_gemini_response(
            litertlm_response, finish_reason="MAX_TOKENS"
        ),
        expected,
    )

  def test_get_engine_caching(self):
    mock_model = mock.MagicMock()
    mock_model.exists.return_value = True
    mock_model.model_path = "/path/to/model"
    mock_model_mod.Model.from_model_id.return_value = mock_model

    mock_engine_instance = mock.MagicMock()
    mock_litert_lm.Engine.return_value = mock_engine_instance

    # First call - creates engine
    engine1 = serve.get_engine("test-model")
    self.assertEqual(engine1, mock_engine_instance)
    mock_litert_lm.Engine.assert_called_once()

    # Second call with same ID - returns cached engine
    engine2 = serve.get_engine("test-model")
    self.assertEqual(engine2, mock_engine_instance)
    self.assertEqual(mock_litert_lm.Engine.call_count, 1)

    # Third call with different ID - replaces engine
    engine3 = serve.get_engine("other-model")
    self.assertEqual(engine3, mock_engine_instance)
    self.assertEqual(mock_litert_lm.Engine.call_count, 2)
    mock_engine_instance.__exit__.assert_called_once()

  def test_get_engine_recovery_after_failure(self):
    # Setup mocks for model "A" (exists) and "B" (missing)
    def from_id_side_effect(model_id):
      m = mock.MagicMock()
      m.exists.return_value = model_id == "A"
      m.model_path = f"/path/to/{model_id}"
      return m

    mock_model_mod.Model.from_model_id.side_effect = from_id_side_effect

    mock_engine_instance = mock.MagicMock()
    mock_litert_lm.Engine.return_value = mock_engine_instance

    # 1. Load model "A"
    serve.get_engine("A")
    self.assertEqual(mock_litert_lm.Engine.call_count, 1)

    # 2. Try to load model "B" (fails)
    with self.assertRaises(FileNotFoundError):
      serve.get_engine("B")

    # The previous engine for "A" should have been exited
    mock_engine_instance.__exit__.assert_called_once()

    # 3. Load model "A" again. It should create a NEW engine.
    serve.get_engine("A")
    self.assertEqual(mock_litert_lm.Engine.call_count, 2)

  def test_model_id_regex_parsing(self):
    # Valid model ID
    match = serve.GEN_CONTENT_RE.match(
        "/v1beta/models/gemma-2b:generateContent"
    )
    self.assertIsNotNone(match)
    self.assertEqual(match.group(1), "gemma-2b")

    # Another valid model ID
    match = serve.STREAM_GEN_CONTENT_RE.match(
        "/v1beta/models/my_model-1:streamGenerateContent"
    )
    self.assertIsNotNone(match)
    self.assertEqual(match.group(1), "my_model-1")

    # Invalid: contains slash
    self.assertIsNone(
        serve.GEN_CONTENT_RE.match("/v1beta/models/gemma/2b:generateContent")
    )

    # Invalid: contains backslash
    self.assertIsNone(
        serve.STREAM_GEN_CONTENT_RE.match(
            "/v1beta/models/gemma\\2b:streamGenerateContent"
        )
    )

    # Invalid: contains colon
    self.assertIsNone(
        serve.GEN_CONTENT_RE.match("/v1beta/models/gemma:2b:generateContent")
    )


if __name__ == "__main__":
  absltest.main()
