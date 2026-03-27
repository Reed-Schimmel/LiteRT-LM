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

"""Main script for litert-lm binary."""

import datetime
import os
import shutil
import subprocess

import fire

import litert_lm
from litert_lm_cli import model
from litert_lm_cli import venv_manager


class LiteRTLMCLI:
  """CLI tool for LiteRT-LM models."""

  def convert(self, source, model_id=None, **kwargs):
    """Converts a HuggingFace model to LiteRT-LM format.

    Args:
      source: The HuggingFace model ID or path (e.g., "google/gemma-2b-it").
      model_id: The ID to store the model as. Defaults to source.
      **kwargs: Additional arguments passed to litert-torch.
    """
    effective_model_id = model_id or source
    if any(
        m.model_id == effective_model_id for m in model.Model.get_all_models()
    ):
      print(f"Error: Model ID '{effective_model_id}' already exists.\n")
      print("Suggestions:")
      print(
          "  1. Run the existing model with 'litert-lm run"
          f" {effective_model_id}'."
      )
      print(
          f"  2. Convert again using 'litert-lm convert {effective_model_id}'"
          " with '--model_id=other-model-id' to set a different model ID for"
          " the converted model."
      )
      print(
          "  3. Rename the existing model with 'litert-lm rename"
          f" {effective_model_id} <new_model_id>' and convert the model again."
      )
      return

    venv_manager.recreate_venv_if_self_managed()
    venv_manager.ensure_binary(venv_manager.LITERT_TORCH_BIN)

    output_dir = model.get_model_dir(effective_model_id)
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        venv_manager.LITERT_TORCH_BIN,
        "export_hf",
        "--model",
        source,
        "--output_dir",
        output_dir,
        "--bundle_litert_lm",
    ]

    for key, value in kwargs.items():
      flag = "--" + key.replace("_", "-")
      if isinstance(value, bool):
        if value:
          cmd.append(flag)
      else:
        cmd.append(flag)
        cmd.append(str(value))

    print(f"Running: {' '.join(cmd)}")
    try:
      subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
      print(f"Error: Model conversion failed with exit code {e.returncode}.")
      print("Check the logs above for the specific error message.")
      return

    print(f"You can now run the model with 'run {effective_model_id}'")

  def list(self):
    """Lists all imported LiteRT-LM models."""
    base_dir = model.get_converted_models_base_dir()
    print(f"Listing models in: {base_dir}")

    models = sorted(model.Model.get_all_models(), key=lambda m: m.model_id)

    # Calculate dynamic width for ID column
    id_width = max([len(m.model_id) for m in models] + [len("ID"), 25]) + 2

    print(f"{'ID':<{id_width}} {'SIZE':<15} {'MODIFIED'}")

    for model_item in models:
      path = model_item.model_path
      try:
        stat = os.stat(path)
        size_bytes = stat.st_size
        if size_bytes >= 1024 * 1024 * 1024:
          size_str = f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"
        else:
          size_str = f"{size_bytes / (1024 * 1024):.1f} MB"
        modified_date = datetime.datetime.fromtimestamp(stat.st_mtime).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
      except FileNotFoundError:
        size_str = "Unknown"
        modified_date = "Unknown"

      print(f"{model_item.model_id:<{id_width}} {size_str:<15} {modified_date}")

  def _import_model(self, source, ref):
    """Imports a model from a local path."""
    if not os.path.exists(source):
      print(f"Source file not found: {source}")
      return

    model_obj = model.Model.from_model_id(ref)
    model_path = model_obj.model_path
    model_dir = os.path.dirname(model_path)

    os.makedirs(model_dir, exist_ok=True)

    shutil.copy(source, model_path)
    print(f"Successfully imported model to {model_path}")

  def delete(self, model_id):
    """Deletes a model from the local storage.

    Args:
      model_id: The ID of the model to delete.
    """
    model_obj = model.Model.from_model_id(model_id)
    model_dir = os.path.dirname(model_obj.model_path)
    if os.path.exists(model_dir) and model_dir.startswith(
        model.get_converted_models_base_dir()
    ):
      shutil.rmtree(model_dir)
      print(f"Deleted model: {model_id}")
    else:
      print(f"Model not found: {model_id}")

  def rename(self, old_model_id, new_model_id):
    """Renames a model.

    Args:
      old_model_id: The current model ID.
      new_model_id: The new model ID.
    """
    old_model = model.Model.from_model_id(old_model_id)
    if not old_model.exists():
      print(f"Model not found: {old_model_id}")
      return

    new_model = model.Model.from_model_id(new_model_id)
    if new_model.exists():
      print(f"Target model ID already exists: {new_model_id}")
      return

    old_dir = os.path.dirname(old_model.model_path)
    new_dir = os.path.dirname(new_model.model_path)

    os.makedirs(os.path.dirname(new_dir), exist_ok=True)
    shutil.move(old_dir, new_dir)
    print(f'Renamed model "{old_model_id}" to "{new_model_id}"')

  def benchmark(
      self,
      model_reference: str,
      prefill_tokens: int = 256,
      decode_tokens: int = 256,
      android: bool = False,
      backend: str = "cpu",
      verbose: bool = False,
  ):
    """Benchmarks a LiteRT-LM model.

    Args:
      model_reference: A relative or absolute path to a .litertlm model file, or
        a model ID from `litert-lm list`.
      prefill_tokens: The number of tokens to prefill.
      decode_tokens: The number of tokens to decode.
      android: Whether to run the benchmark on an Android device via ADB.
      backend: The backend to use (cpu or gpu).
      verbose: Whether to enable verbose logging.
    """
    if verbose:
      litert_lm.set_min_log_severity(litert_lm.LogSeverity.VERBOSE)

    model_obj = model.Model.from_model_reference(model_reference)
    model_obj.benchmark(
        prefill_tokens=prefill_tokens,
        decode_tokens=decode_tokens,
        is_android=android,
        backend=backend,
    )

  def run(
      self,
      model_reference,
      prompt=None,
      android=False,
      backend="cpu",
      preset=None,
      verbose=False,
  ):
    r'''Runs a LiteRT-LM model interactively or with a single prompt.

    Example preset file:
      ```py
      def add_numbers(a: float, b: float) -> float:
        """Adds two numbers."""
        return a + b

      # Provides the "system instruction", "tools", and "extra_context"
      system_instruction = "You are a helpful assistant."
      tools = [add_numbers]
      extra_context = {"key": "value"}
      ```

    Args:
      model_reference: A relative or absolute path to a .litertlm model file, or
        a model ID from `litert-lm list`. If the model is not found locally and
        the reference looks like a HuggingFace repository ID (e.g.,
        "google/gemma-3-1b-it"), an automatic conversion will be attempted.
      prompt: A single prompt to run once and exit.
      android: Whether to run the model on an Android device via ADB.
      backend: The backend to use (cpu or gpu).
      preset: Path to a Python file containing tool functions and system
        instructions.
      verbose: Whether to enable verbose logging.
    '''
    if verbose:
      litert_lm.set_min_log_severity(litert_lm.LogSeverity.VERBOSE)

    model_obj = model.Model.from_model_reference(model_reference)
    if not model_obj.exists():
      # Only auto-convert if it looks like a HuggingFace repo ID (account/repo)
      # and is not a local path.
      parts = model_reference.split("/")
      if len(parts) == 2 and all(parts) and not os.path.exists(model_reference):
        print(
            f"Model '{model_reference}' not found. Attempting to convert from"
            f" https://huggingface.co/{model_reference} ..."
        )
        self.convert(model_reference)
        model_obj = model.Model.from_model_reference(model_reference)

      if not model_obj.exists():
        print(f"Failed to find or convert model '{model_reference}'.")
        return

    model_obj.run_interactive(
        prompt=prompt, is_android=android, backend=backend, preset=preset
    )


# "import" is a reserved keyword in Python and cannot be used as a function name
# directly. Instead, use setattr to create the sub-command import.
setattr(LiteRTLMCLI, "import", LiteRTLMCLI._import_model)  # pylint: disable=protected-access


def main():
  litert_lm.set_min_log_severity(litert_lm.LogSeverity.ERROR)
  fire.Fire(LiteRTLMCLI())


if __name__ == "__main__":
  main()
