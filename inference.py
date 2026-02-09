import argparse
from pathlib import Path

import torch
import torchaudio
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    WhisperFeatureExtractor,
)

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


def get_audio_token_length(seconds, merge_factor=2):
    def get_T_after_cnn(L_in, dilation=1):
        # CNN layer configurations: (padding, kernel_size, stride)
        layer_configs = [(1, 3, 1), (1, 3, 2)]
        for padding, kernel_size, stride in layer_configs:
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

    # Check if audio is empty
    if wav.shape[1] == 0:
        raise ValueError(f"Audio file is empty or has no samples: {audio_path}")

    tokens = []
    tokens += tokenizer.encode("<|user|>")
    tokens += tokenizer.encode("\n")

    audios = []
    audio_offsets = []
    audio_length = []
    chunk_size = chunk_seconds * feature_extractor.sampling_rate
    for start in range(0, wav.shape[1], chunk_size):
        chunk = wav[:, start : start + chunk_size]
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


def prepare_inputs(batch: dict, device: torch.device) -> tuple[dict, int]:
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
    args = parser.parse_args()

    transcribe(
        checkpoint_dir=Path(args.checkpoint_dir),
        audio_path=Path(args.audio),
        tokenizer_path=args.tokenizer_path,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
    )


if __name__ == "__main__":
    main()
