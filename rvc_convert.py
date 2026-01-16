#!/usr/bin/env python3
"""
Simple voice conversion script that avoids WebUI crashes on macOS
Usage: python convert_voice.py --input /path/to/audio.wav --output /path/to/output.wav
"""
import os
import sys

# Set environment variables before imports
os.environ["FORCE_CPU_MODE"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

now_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(now_dir)

from dotenv import load_dotenv
from scipy.io import wavfile
from configs.config import Config
from infer.modules.vc.modules import VC

load_dotenv()


def convert_voice(
    input_path,
    output_path,
    model_name="kamel's voice",
    index_path=None,
    f0method="pm",  # Use pm for speed, or rmvpe for quality
    f0up_key=0,
    index_rate=0.75,
    filter_radius=3,
    resample_sr=0,
    rms_mix_rate=1,
    protect=0.33,
):
    """Convert voice using command-line interface"""
    print(f"Loading model: {model_name}")
    config = Config()
    # Force CPU mode to avoid MPS crashes
    if config.device == "mps":
        config.device = "cpu"
        config.instead = "cpu"
        config.is_half = False

    vc = VC(config)
    # Remove .pth extension if present (get_vc expects name without extension)
    if model_name.endswith(".pth"):
        model_name = model_name[:-4]
    vc.get_vc(model_name)

    print(f"Converting: {input_path}")
    info, wav_opt = vc.vc_single(
        0,  # speaker ID
        input_path,
        f0up_key,
        None,  # f0_file
        f0method,
        index_path,
        None,  # file_index2
        index_rate,
        filter_radius,
        resample_sr,
        rms_mix_rate,
        protect,
    )

    if "Success" in info:
        print(f"Saving to: {output_path}")
        wavfile.write(output_path, wav_opt[0], wav_opt[1])
        print("✓ Conversion successful!")
        print(info)
        return True
    else:
        print("✗ Conversion failed:")
        print(info)
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Voice conversion without WebUI")
    parser.add_argument("--input", required=True, help="Input audio file path")
    parser.add_argument("--output", required=True, help="Output audio file path")
    parser.add_argument(
        "--model", default="kamel's voice", help="Model name (without .pth extension)"
    )
    parser.add_argument("--index", default=None, help="Index file path (optional)")
    parser.add_argument(
        "--f0method",
        default="pm",
        choices=["pm", "harvest", "rmvpe"],
        help="F0 extraction method",
    )
    parser.add_argument(
        "--index_rate", type=float, default=0.75, help="Index rate (0.0-1.0)"
    )
    parser.add_argument(
        "--f0up_key", type=int, default=0, help="Pitch shift (semitones)"
    )

    args = parser.parse_args()

    convert_voice(
        args.input,
        args.output,
        model_name=args.model,
        index_path=args.index,
        f0method=args.f0method,
        f0up_key=args.f0up_key,
        index_rate=args.index_rate,
    )
