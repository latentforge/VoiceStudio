# Copyright 2025 SparkAudio, Xinsheng Wang and the LatentForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch Spark-TTS model."""

from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM, Qwen2PreTrainedModel
from transformers.utils import auto_docstring

from .configuration_spark_tts import SparkTTSConfig
from .weight_conversion import convert_published_checkpoint


@auto_docstring
class SparkTTSPreTrainedModel(Qwen2PreTrainedModel):
    config: SparkTTSConfig


@auto_docstring(
    custom_intro="""
    Spark-TTS, a Qwen2 decoder whose vocabulary is extended with BiCodec semantic and global tokens plus task and
    style control tokens, so that speech synthesis, voice cloning and attribute control are all next-token prediction
    over one flat sequence. Reference audio is turned into global tokens and generated semantic tokens are turned
    back into a waveform by [`SparkTTSBiCodecModel`], which [`SparkTTSProcessor`] holds.

    The training objective is the one [`Qwen2ForCausalLM`] computes: cross entropy of the next token over the joint
    text/BiCodec vocabulary, on the positions `labels` leaves unmasked.
    """
)
class SparkTTSForConditionalGeneration(SparkTTSPreTrainedModel, Qwen2ForCausalLM):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r"""
        Load a Spark-TTS model, from either a converted checkpoint or the published `SparkAudio/Spark-TTS-0.5B`
        layout, which is three independently saved models in three subfolders plus two YAML files and which is
        converted on first use.

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                A Hugging Face repo id or a local directory.
            model_args (`tuple`, *optional*):
                Forwarded to [`~PreTrainedModel.from_pretrained`].
            kwargs (`dict[str, Any]`, *optional*):
                Forwarded to [`~PreTrainedModel.from_pretrained`].

        Returns:
            [`SparkTTSForConditionalGeneration`]: The loaded model.

        Raises:
            OSError: If the checkpoint holds neither the weights [`~PreTrainedModel.from_pretrained`] expects nor
                the published Spark-TTS layout.
        """
        try:
            return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        except OSError:
            converted = convert_published_checkpoint(pretrained_model_name_or_path, **kwargs)
            if converted is None:
                raise
        return super().from_pretrained(converted, *model_args, **kwargs)


__all__ = ["SparkTTSForConditionalGeneration", "SparkTTSPreTrainedModel"]
