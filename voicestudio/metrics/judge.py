"""The models a metric asks about an artifact, and where they run."""

import base64
import json
import logging
import socket
import subprocess
import time
import urllib.error
import urllib.request
from abc import ABC
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from ..utils.audio_utils import load_audio
from .base import LauncherType, MetricConfig


logger = logging.getLogger(__name__)


class ChatCompletionClient:
    """Reaches a model over the OpenAI compatible chat completions route.

    Built on the standard library rather than an HTTP package, since none is declared here and a
    judge posting one request per artifact does not need a connection pool to work. What it does
    need is to be told when a service is not answering rather than to hang, which is what the
    timeout and the retries are for.

    Args:
        base_url (`str`):
            Where the service answers, with or without the trailing `/v1`.
        model_id (`str`, *optional*):
            The model the service is asked for, which a server holding one still wants named.
        api_key (`str`, *optional*):
            Sent as a bearer token. A locally served model usually needs none.
        timeout (`float`, *optional*, defaults to 120.0):
            Seconds to wait on one request. A judge listening to a long clip is slow, and a
            timeout tuned for text will read as a service that is down.
        retries (`int`, *optional*, defaults to 2):
            Attempts after the first, on a transport error or a 5xx. A judged benchmark is
            thousands of requests and one dropped connection should not cost the run.
    """

    def __init__(
        self,
        base_url: str,
        model_id: str | None = None,
        api_key: str | None = None,
        timeout: float = 120.0,
        retries: int = 2,
    ):
        self.base_url = base_url.rstrip("/")
        if not self.base_url.endswith("/v1"):
            self.base_url = f"{self.base_url}/v1"
        self.model_id = model_id
        self.api_key = api_key
        self.timeout = timeout
        self.retries = retries

    def complete(self, messages: list[dict[str, Any]], **parameters: Any) -> dict[str, Any]:
        """Sends one conversation and returns what came back.

        Args:
            messages (`list[dict[str, Any]]`):
                The conversation, in the OpenAI content shape. An audio part is
                `{"type": "input_audio", "input_audio": {"data": <base64>, "format": "wav"}}`.
            **parameters:
                Decoding settings passed through, such as `temperature` or `max_tokens`. What was
                sent belongs beside the answer, since two runs decoded differently are not
                comparable.

        Returns:
            `dict[str, Any]`: The decoded response body.

        Raises:
            RuntimeError: If the service did not answer after every attempt.
        """
        payload = {"messages": messages, **parameters}
        if self.model_id:
            payload["model"] = self.model_id
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        last: Exception | None = None
        for attempt in range(self.retries + 1):
            request = urllib.request.Request(
                f"{self.base_url}/chat/completions", data=body, headers=headers
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as error:
                # A 4xx is the request being wrong and will be wrong again; only a 5xx is worth
                # asking twice.
                if error.code < 500:
                    raise
                last = error
            except (urllib.error.URLError, TimeoutError, OSError) as error:
                last = error
            if attempt < self.retries:
                time.sleep(2**attempt)
        raise RuntimeError(f"{self.base_url} did not answer after {self.retries + 1} attempts") from last


class VllmServer:
    """A vLLM process brought up to serve one model, and stopped again afterwards.

    Started through `uv tool run` so the server resolves its own environment, which is what keeps
    a judge's answers from moving when the training environment does.

    Args:
        model_id (`str`):
            Repository id of the model to serve.
        revision (`str`, *optional*):
            Which commit of it. A moved tag serves a different model under the same name.
        port (`int`, *optional*):
            Port to listen on. Left out, a free one is taken, so two runs on one host do not
            collide.
        startup_timeout (`float`, *optional*, defaults to 600.0):
            Seconds to wait for the service to answer. Loading a quantized MoE off cold storage is
            minutes, not seconds.
        extra_args (`list[str]`, *optional*):
            Passed to `vllm serve` as they are, such as a quantization or a tensor parallel size.

    Raises:
        RuntimeError: If the process exits, or does not answer within `startup_timeout`.
    """

    def __init__(
        self,
        model_id: str,
        revision: str | None = None,
        port: int | None = None,
        startup_timeout: float = 600.0,
        extra_args: list[str] | None = None,
    ):
        self.model_id = model_id
        self.port = port or _free_port()
        self.base_url = f"http://127.0.0.1:{self.port}/v1"

        command = ["uv", "tool", "run", "--from", "vllm", "vllm", "serve", model_id]
        command += ["--port", str(self.port)]
        if revision:
            command += ["--revision", revision]
        command += list(extra_args or [])

        logger.info("starting vllm: %s", " ".join(command))
        self.process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        self._await_ready(startup_timeout)

    def _await_ready(self, timeout: float) -> None:
        """Waits until the service answers, or gives up and takes the process with it.

        Args:
            timeout (`float`):
                Seconds to wait.

        Raises:
            RuntimeError: If the process exited, or the deadline passed.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"vllm exited with {self.process.returncode} before serving {self.model_id}"
                )
            try:
                with urllib.request.urlopen(f"{self.base_url}/models", timeout=5):
                    logger.info("vllm serving %s on port %d", self.model_id, self.port)
                    return
            except (urllib.error.URLError, TimeoutError, OSError):
                time.sleep(2.0)
        self.stop()
        raise RuntimeError(f"vllm did not serve {self.model_id} within {timeout:.0f}s")

    def stop(self) -> None:
        """Stops the process, and kills it if it will not stop."""
        if self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()


def _free_port() -> int:
    """Takes a port the operating system says is free.

    Returns:
        `int`: The port. It is released again before the server binds it, so two servers started
        in the same instant can still collide.
    """
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])

class Judge(ABC):
    """A model a metric asks about an artifact, brought up wherever `launcher` says it lives.

    The three cases live here rather than in a metric, because where a model runs is the
    benchmark's decision and the arithmetic over its answers is the metric's: an error rate should
    not become three classes because a recognizer moved behind a service. A subclass supplies
    [`Judge.load_model`] for the in process case and reads [`Judge.client`] for the other two.

    Args:
        config (`MetricConfig`):
            Which model, where it runs, and how it is reached.
    """

    def __init__(self, config: MetricConfig):
        self.config = config
        self.model: Any = None
        self.client: ChatCompletionClient | None = None
        self._server: VllmServer | None = None

    @property
    def model_id(self) -> str | None:
        """Which checkpoint answers, recorded beside every reading it produced."""
        return self.config.model_id

    def load(self) -> None:
        """Brings the model up, once, wherever `launcher` says it lives.

        Raises:
            ValueError: If a remote launcher was named with no endpoint to reach.
        """
        if self.config.launcher is LauncherType.INPROCESS:
            if self.model is None:
                self.model = self.load_model()
            return
        if self.config.launcher is LauncherType.VLLM and self._server is None:
            self._server = self.start_server()
        if self.client is None:
            endpoint = self.config.endpoint or (self._server.base_url if self._server else None)
            if not endpoint:
                raise ValueError(
                    f"{type(self).__name__} runs under {self.config.launcher} and names no endpoint"
                )
            self.client = ChatCompletionClient(
                endpoint,
                model_id=self.config.model_id,
                **self.config.options.get("client", {}),
            )

    def release(self) -> None:
        """Drops whatever [`Judge.load`] brought up, and frees what it held."""
        if self.model is not None:
            self.release_model()
            self.model = None
        self.client = None
        if self._server is not None:
            self._server.stop()
            self._server = None

    def __enter__(self) -> "Judge":
        """Loads the model, so a caller can hold one at a time rather than all of them."""
        self.load()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        """Releases the model."""
        self.release()

    def load_model(self) -> Any:
        """Builds the model on this process's device.

        Called only under `LauncherType.INPROCESS`.

        Returns:
            `Any`: The model, kept on [`Judge.model`].
        """
        return None

    def release_model(self) -> None:
        """Frees what [`Judge.load_model`] built, before the handle is dropped."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def start_server(self) -> VllmServer:
        """Brings up the vLLM process this model is reached through.

        Returns:
            `VllmServer`: The running server, stopped again by [`Judge.release`].

        Raises:
            ValueError: If the configuration names no `model_id` to serve.
        """
        if not self.config.model_id:
            raise ValueError(f"{type(self).__name__} runs under vllm and names no `model_id`")
        return VllmServer(
            self.config.model_id,
            revision=self.config.revision,
            **self.config.options.get("vllm", {}),
        )


class Transcriber(Judge):
    """Turns paths to generated audio into text with Whisper, for a metric that scores transcripts.

    Held by the metrics that need one rather than inherited into them, so scoring a word rate and a
    character rate over one run loads one recognizer instead of two, and so the caller decides when
    it comes off the card.

    Args:
        config (`MetricConfig`):
            Which recognizer, at what precision, on which device. `model_id` may be a bare size
            such as `"large-v3"`, which names an OpenAI release. `options["language"]` is passed to
            `generate`; letting Whisper detect it instead makes the transcript depend on the audio
            being scored, which is the thing under test.
    """

    def __init__(self, config: MetricConfig):
        super().__init__(config)
        if not config.model_id:
            raise ValueError("a transcriber needs a `model_id` to transcribe with")
        # A bare size such as "large-v3" names an OpenAI release; anything with an owner is taken as is.
        self.checkpoint = (
            config.model_id if "/" in config.model_id else f"openai/whisper-{config.model_id}"
        )
        self.language = config.options.get("language", "en")
        self.device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = getattr(torch, config.dtype) if isinstance(config.dtype, str) else torch.float16

    @property
    def model_id(self) -> str:
        """The checkpoint that produced a transcript, recorded beside every rate it fed."""
        return self.checkpoint

    def load_model(self) -> Any:
        """Loads the Whisper checkpoint and its processor.

        Returns:
            `tuple`: The processor and the model, read in that order.
        """
        processor = WhisperProcessor.from_pretrained(self.checkpoint)
        model = WhisperForConditionalGeneration.from_pretrained(self.checkpoint, dtype=self.dtype)
        return processor, model.to(self.device).eval()

    def transcribe(self, audio_paths: Sequence[str]) -> list[str]:
        """Transcribes generated audio in batches.

        Args:
            audio_paths (`Sequence[str]`):
                Paths to the generated audio, one per utterance.

        Returns:
            `list[str]`: One transcript per input path, in input order.

        Raises:
            NotImplementedError: If a remote launcher was named. Serving a recognizer answers on
                the transcription route rather than the chat one, which is not written here.
        """
        if self.config.launcher is not LauncherType.INPROCESS:
            raise NotImplementedError(
                f"transcribing under {self.config.launcher} needs the audio transcription route, "
                "which this client does not speak"
            )
        self.load()
        processor, model = self.model
        sampling_rate = processor.feature_extractor.sampling_rate

        # A path repeated across utterances is transcribed once. Whisper dominates the runtime here.
        unique_paths = list(dict.fromkeys(audio_paths))
        transcripts = {}
        for start in range(0, len(unique_paths), self.config.batch_size):
            batch = unique_paths[start : start + self.config.batch_size]
            waveforms = [load_audio(path, sampling_rate).numpy() for path in batch]
            # Whisper's pad token is its eos token, so a batch without an explicit mask decodes
            # against padding it cannot identify.
            features = processor(
                waveforms, sampling_rate=sampling_rate, return_tensors="pt", return_attention_mask=True
            )
            with torch.no_grad():
                predicted_ids = model.generate(
                    features.input_features.to(device=self.device, dtype=model.dtype),
                    attention_mask=features.attention_mask.to(self.device),
                    language=self.language,
                    task="transcribe",
                )
            decoded = processor.batch_decode(predicted_ids, skip_special_tokens=True)
            transcripts.update(zip(batch, (text.strip() for text in decoded)))

        return [transcripts[path] for path in audio_paths]


class Qwen3OmniJudge(Judge):
    """Asks Qwen3-Omni what it heard, for the benchmarks whose answer is an opinion.

    The published InstructTTSEval and EmergentTTS-Eval both judged with a Gemini model that has
    since been retired, so a benchmark scored by this one is a version of its own rather than the
    published number. That substitution is why the model is named on every reading.

    Args:
        config (`MetricConfig`):
            Which checkpoint, where it runs, and how it is reached. `options["generation"]` holds
            the decoding settings, which belong beside the answers: two runs decoded differently
            are not comparable and nothing downstream would notice.
    """

    DEFAULT_MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    """The checkpoint used when the configuration names none."""

    def load_model(self) -> Any:
        """Loads the checkpoint and its processor onto this process's device.

        Returns:
            `tuple`: The processor and the model, which [`Qwen3OmniJudge.ask`] reads in that order.
        """
        from transformers import AutoProcessor, Qwen3OmniMoeForConditionalGeneration

        model_id = self.config.model_id or self.DEFAULT_MODEL_ID
        processor = AutoProcessor.from_pretrained(model_id, revision=self.config.revision)
        model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            model_id,
            revision=self.config.revision,
            dtype=self.config.dtype or "auto",
            device_map=self.config.device or "auto",
        )
        return processor, model.eval()

    def ask(self, audio_path: str, prompt: str, **generation: Any) -> str:
        """Plays one clip to the judge and returns what it answered.

        Args:
            audio_path (`str`):
                The clip to play.
            prompt (`str`):
                What the judge is asked about it.
            **generation:
                Decoding settings, merged over `options["generation"]`.

        Returns:
            `str`: The answer, stripped.

        Raises:
            RuntimeError: If the judge was not loaded first.
        """
        settings = {**self.config.options.get("generation", {}), **generation}
        if self.client is not None:
            return self._ask_remote(audio_path, prompt, settings)
        if self.model is None:
            raise RuntimeError(f"{type(self).__name__} was asked before it was loaded")
        return self._ask_local(audio_path, prompt, settings)

    def _ask_remote(self, audio_path: str, prompt: str, settings: dict[str, Any]) -> str:
        """Posts the clip to the service as an OpenAI audio part.

        Args:
            audio_path (`str`):
                The clip to send.
            prompt (`str`):
                What the judge is asked.
            settings (`dict[str, Any]`):
                Decoding settings.

        Returns:
            `str`: The answer.
        """
        with open(audio_path, "rb") as handle:
            encoded = base64.b64encode(handle.read()).decode("ascii")
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {"data": encoded, "format": Path(audio_path).suffix.lstrip(".") or "wav"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        response = self.client.complete(messages, **settings)
        return response["choices"][0]["message"]["content"].strip()

    def _ask_local(self, audio_path: str, prompt: str, settings: dict[str, Any]) -> str:
        """Runs the clip through the loaded checkpoint.

        Args:
            audio_path (`str`):
                The clip to play.
            prompt (`str`):
                What the judge is asked.
            settings (`dict[str, Any]`):
                Decoding settings.

        Returns:
            `str`: The answer.
        """
        processor, model = self.model
        sampling_rate = processor.feature_extractor.sampling_rate
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        inputs = processor(
            text=text,
            audio=[load_audio(audio_path, sampling_rate).numpy()],
            sampling_rate=sampling_rate,
            return_tensors="pt",
        ).to(model.device)
        with torch.no_grad():
            generated = model.generate(**inputs, **settings)
        answered = generated[:, inputs["input_ids"].shape[-1] :]
        return processor.batch_decode(answered, skip_special_tokens=True)[0].strip()


__all__ = ["ChatCompletionClient", "Judge", "Qwen3OmniJudge", "Transcriber", "VllmServer"]
