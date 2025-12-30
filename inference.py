import argparse
from pathlib import Path
from typing import Generator

import torch
import torchaudio
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          WhisperFeatureExtractor)

WHISPER_FEAT_CFG = {
    "chunk_length": 30,
    "feature_extractor_type": "WhisperFeatureExtractor",
    "feature_size": 128,
    "hop_length": 160,
    "n_fft": 400,
    "n_samples": 480000,
    "nb_max_frames": 3000,
    "padding_side": "right",
    "padding_value": 0.0,
    "processor_class": "WhisperProcessor",
    "return_attention_mask": False,
    "sampling_rate": 16000,
}


def chunk_iter(
    inputs: torch.Tensor,
    chunk_len: int,
    stride_left: int,
    stride_right: int,
) -> Generator[dict, None, None]:
    """
    Iterate over audio in overlapping chunks following HuggingFace Transformers spec.

    Args:
        inputs: Audio tensor of shape (1, samples) or (samples,)
        chunk_len: Number of samples per chunk
        stride_left: Number of overlap samples on left side
        stride_right: Number of overlap samples on right side

    Yields:
        dict with keys:
            - chunk: Audio chunk tensor
            - stride: Tuple of (chunk_length, left_stride, right_stride)
            - is_last: Whether this is the final chunk
            - start_sample: Starting sample index in original audio
    """
    if inputs.dim() == 2:
        inputs = inputs.squeeze(0)

    inputs_len = inputs.shape[0]
    step = chunk_len - stride_left - stride_right

    for chunk_start_idx in range(0, inputs_len, step):
        chunk_end_idx = chunk_start_idx + chunk_len
        chunk = inputs[chunk_start_idx:chunk_end_idx]

        # First chunk: no left stride
        _stride_left = 0 if chunk_start_idx == 0 else stride_left
        # Last chunk: no right stride
        is_last = chunk_end_idx >= inputs_len
        _stride_right = 0 if is_last else stride_right

        # Skip if chunk is too small (only stride content)
        if chunk.shape[0] > _stride_left:
            yield {
                "chunk": chunk.unsqueeze(0),  # (1, samples)
                "stride": (chunk.shape[0], _stride_left, _stride_right),
                "is_last": is_last,
                "start_sample": chunk_start_idx,
            }

        if is_last:
            break


def get_audio_token_length(seconds, merge_factor=2):
    def get_T_after_cnn(L_in, dilation=1):
        L_out = L_in
        for padding, kernel_size, stride in eval("[(1,3,1)] + [(1,3,2)] "):
            L_out = L_in + 2 * padding - dilation * (kernel_size - 1) - 1
            L_out = 1 + L_out // stride
            L_in = L_out
        return L_out

    mel_len = int(seconds * 100)
    audio_len_after_cnn = get_T_after_cnn(mel_len)
    audio_token_num = (audio_len_after_cnn - merge_factor) // merge_factor + 1

    # TODO: current whisper model can't process longer sequence, maybe cut chunk in the future
    audio_token_num = min(audio_token_num, 1500 // merge_factor)

    return audio_token_num


def build_prompt(
    audio_path: Path,
    tokenizer,
    feature_extractor: WhisperFeatureExtractor,
    merge_factor: int,
    chunk_seconds: int = 30,
) -> dict:
    audio_path = Path(audio_path)
    wav, sr = torchaudio.load(str(audio_path))
    wav = wav[:1, :]
    if sr != feature_extractor.sampling_rate:
        wav = torchaudio.transforms.Resample(sr, feature_extractor.sampling_rate)(wav)

    tokens = []
    tokens += tokenizer.encode("<|user|>")
    tokens += tokenizer.encode("\n")

    audios = []
    audio_offsets = []
    audio_length = []
    chunk_size = chunk_seconds * feature_extractor.sampling_rate
    for start in range(0, wav.shape[1], chunk_size):
        chunk = wav[:, start: start + chunk_size]
        mel = feature_extractor(
            chunk.numpy(),
            sampling_rate=feature_extractor.sampling_rate,
            return_tensors="pt",
            padding="max_length",
        )["input_features"]
        audios.append(mel)
        seconds = chunk.shape[1] / feature_extractor.sampling_rate
        num_tokens = get_audio_token_length(seconds, merge_factor)
        tokens += tokenizer.encode("<|begin_of_audio|>")
        audio_offsets.append(len(tokens))
        tokens += [0] * num_tokens
        tokens += tokenizer.encode("<|end_of_audio|>")
        audio_length.append(num_tokens)

    if not audios:
        raise ValueError("音频内容为空或加载失败。")

    tokens += tokenizer.encode("<|user|>")
    tokens += tokenizer.encode("\nPlease transcribe this audio into text")

    tokens += tokenizer.encode("<|assistant|>")
    tokens += tokenizer.encode("\n")

    batch = {
        "input_ids": torch.tensor([tokens], dtype=torch.long),
        "audios": torch.cat(audios, dim=0),
        "audio_offsets": [audio_offsets],
        "audio_length": [audio_length],
        "attention_mask": torch.ones(1, len(tokens), dtype=torch.long),
    }
    return batch


def build_single_chunk_prompt(
    chunk: torch.Tensor,
    tokenizer,
    feature_extractor: WhisperFeatureExtractor,
    merge_factor: int,
) -> dict:
    """Build prompt for a single audio chunk."""
    mel = feature_extractor(
        chunk.numpy(),
        sampling_rate=feature_extractor.sampling_rate,
        return_tensors="pt",
        padding="max_length",
    )["input_features"]

    seconds = chunk.shape[1] / feature_extractor.sampling_rate
    num_tokens = get_audio_token_length(seconds, merge_factor)

    tokens = []
    tokens += tokenizer.encode("<|user|>")
    tokens += tokenizer.encode("\n")
    tokens += tokenizer.encode("<|begin_of_audio|>")
    audio_offset = len(tokens)
    tokens += [0] * num_tokens
    tokens += tokenizer.encode("<|end_of_audio|>")
    tokens += tokenizer.encode("<|user|>")
    tokens += tokenizer.encode("\nPlease transcribe this audio into text")
    tokens += tokenizer.encode("<|assistant|>")
    tokens += tokenizer.encode("\n")

    return {
        "input_ids": torch.tensor([tokens], dtype=torch.long),
        "audios": mel,
        "audio_offsets": [[audio_offset]],
        "audio_length": [[num_tokens]],
        "attention_mask": torch.ones(1, len(tokens), dtype=torch.long),
    }


def prepare_inputs(batch: dict, device: str | torch.device) -> tuple[dict, int]:
    tokens = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    audios = batch["audios"].to(device)
    model_inputs = {
        "inputs": tokens,
        "attention_mask": attention_mask,
        "audios": audios.to(torch.bfloat16),
        "audio_offsets": batch["audio_offsets"],
        "audio_length": batch["audio_length"],
    }
    return model_inputs, tokens.size(1)


def transcribe(
    checkpoint_dir: Path,
    audio_path: Path,
    tokenizer_path: str,
    max_new_tokens: int,
    device: str,
):
    tokenizer_source = tokenizer_path if tokenizer_path else checkpoint_dir
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    feature_extractor = WhisperFeatureExtractor(**WHISPER_FEAT_CFG)

    config = AutoConfig.from_pretrained(checkpoint_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_dir,
        config=config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    batch = build_prompt(
        audio_path,
        tokenizer,
        feature_extractor,
        merge_factor=config.merge_factor,
    )

    model_inputs, prompt_len = prepare_inputs(batch, device)

    with torch.inference_mode():
        generated = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    transcript_ids = generated[0, prompt_len:].cpu().tolist()
    transcript = tokenizer.decode(transcript_ids, skip_special_tokens=True).strip()
    print("----------")
    print(transcript or "[Empty transcription]")


def transcribe_sliding_window(
    checkpoint_dir: Path,
    audio_path: Path,
    tokenizer_path: str,
    max_new_tokens: int,
    device: str,
    chunk_length_s: float = 30.0,
    stride_length_s: float | None = None,
):
    """
    Transcribe long audio using sliding window approach.

    Following HuggingFace Transformers spec:
    - Audio is split into overlapping chunks
    - Each chunk is transcribed independently
    - Transcriptions are concatenated (stride regions provide context but are not double-transcribed)

    Args:
        checkpoint_dir: Path to model checkpoint
        audio_path: Path to audio file
        tokenizer_path: Path to tokenizer (defaults to checkpoint_dir)
        max_new_tokens: Maximum tokens to generate per chunk
        device: Device to run inference on
        chunk_length_s: Length of each chunk in seconds (default: 30)
        stride_length_s: Overlap on each side in seconds (default: chunk_length_s / 6)
    """
    # Default stride: no overlap to avoid duplicate transcriptions
    if stride_length_s is None:
        stride_length_s = 0.0

    tokenizer_source = tokenizer_path if tokenizer_path else checkpoint_dir
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    feature_extractor = WhisperFeatureExtractor(**WHISPER_FEAT_CFG)

    config = AutoConfig.from_pretrained(checkpoint_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_dir,
        config=config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    # Load audio
    audio_path = Path(audio_path)
    wav, sr = torchaudio.load(str(audio_path))
    wav = wav[:1, :]  # mono
    if sr != feature_extractor.sampling_rate:
        wav = torchaudio.transforms.Resample(sr, feature_extractor.sampling_rate)(wav)

    # Calculate chunk parameters in samples
    sampling_rate = feature_extractor.sampling_rate
    chunk_len = int(chunk_length_s * sampling_rate)
    stride_left = int(stride_length_s * sampling_rate)
    stride_right = int(stride_length_s * sampling_rate)

    audio_duration = wav.shape[1] / sampling_rate
    print(f"Audio duration: {audio_duration:.1f}s")
    print(f"Chunk length: {chunk_length_s}s, Stride: {stride_length_s}s")
    print(f"Step size: {chunk_length_s - 2 * stride_length_s}s")
    print("----------")

    transcripts = []
    chunk_idx = 0

    for chunk_data in chunk_iter(wav, chunk_len, stride_left, stride_right):
        chunk = chunk_data["chunk"]
        stride_info = chunk_data["stride"]
        is_last = chunk_data["is_last"]
        start_sample = chunk_data["start_sample"]

        start_time = start_sample / sampling_rate
        chunk_duration = chunk.shape[1] / sampling_rate

        print(f"Processing chunk {chunk_idx + 1}: {start_time:.1f}s - {start_time + chunk_duration:.1f}s")

        # Build prompt for this chunk
        batch = build_single_chunk_prompt(
            chunk,
            tokenizer,
            feature_extractor,
            merge_factor=config.merge_factor,
        )

        model_inputs, prompt_len = prepare_inputs(batch, device)

        with torch.inference_mode():
            generated = model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        transcript_ids = generated[0, prompt_len:].cpu().tolist()
        transcript = tokenizer.decode(transcript_ids, skip_special_tokens=True).strip()

        if transcript:
            transcripts.append(transcript)
            print(f"  -> {transcript[:80]}{'...' if len(transcript) > 80 else ''}")

        chunk_idx += 1

    # Combine all transcriptions
    full_transcript = " ".join(transcripts)
    print("----------")
    print("Full transcription:")
    print(full_transcript or "[Empty transcription]")

    return full_transcript


def main():
    parser = argparse.ArgumentParser(description="Minimal ASR transcription demo.")
    parser.add_argument(
        "--checkpoint_dir", type=str, default=str(Path(__file__).parent)
    )
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file.")
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Tokenizer directory (defaults to checkpoint dir when omitted).",
    )
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--sliding_window",
        action="store_true",
        help="Use sliding window for long audio transcription.",
    )
    parser.add_argument(
        "--chunk_length_s",
        type=float,
        default=30.0,
        help="Chunk length in seconds for sliding window mode (default: 30).",
    )
    parser.add_argument(
        "--stride_length_s",
        type=float,
        default=None,
        help="Stride/overlap length in seconds (default: chunk_length_s / 6).",
    )
    args = parser.parse_args()

    if args.sliding_window:
        transcribe_sliding_window(
            checkpoint_dir=Path(args.checkpoint_dir),
            audio_path=Path(args.audio),
            tokenizer_path=args.tokenizer_path,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
            chunk_length_s=args.chunk_length_s,
            stride_length_s=args.stride_length_s,
        )
    else:
        transcribe(
            checkpoint_dir=Path(args.checkpoint_dir),
            audio_path=Path(args.audio),
            tokenizer_path=args.tokenizer_path,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
        )


if __name__ == "__main__":
    main()
