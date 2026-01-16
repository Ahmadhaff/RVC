# RVC - Retrieval-based Voice Conversion

A powerful and user-friendly voice conversion framework based on VITS, designed for high-quality voice cloning and conversion.

## 🎯 Features

- **High-Quality Voice Conversion**: Convert voices with exceptional quality using retrieval-based feature replacement
- **Fast Training**: Train models quickly even on modest hardware
- **Low Data Requirements**: Achieve good results with as little as 10 minutes of clean audio data
- **Web Interface**: Simple and intuitive web-based UI for all operations
- **Model Fusion**: Change voice characteristics through model merging
- **Vocal Separation**: Integrated UVR5 model for quick vocal/instrumental separation
- **Advanced Pitch Extraction**: Uses state-of-the-art RMVPE algorithm for pitch extraction
- **Cross-Platform**: Supports Windows, Linux, and macOS
- **GPU Acceleration**: Optimized for NVIDIA, AMD, and Intel GPUs

## 📋 Requirements

- Python 3.8 or higher
- PyTorch (see installation instructions below)
- FFmpeg
- Sufficient disk space for models and datasets

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/Ahmadhaff/RVC.git
cd RVC
```

### 2. Install Dependencies

#### Install PyTorch

First, install PyTorch according to your system:

**NVIDIA GPU (CUDA):**
```bash
pip install torch torchvision torchaudio
```

**CPU only:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**macOS (Apple Silicon):**
```bash
pip install torch torchvision torchaudio
```

#### Install Project Dependencies

**NVIDIA GPU:**
```bash
pip install -r requirements.txt
```

**AMD/Intel GPU:**
```bash
pip install -r requirements-dml.txt
```

**AMD ROCm (Linux only):**
```bash
pip install -r requirements-amd.txt
```

**Intel IPEX (Linux only):**
```bash
pip install -r requirements-ipex.txt
```

### 3. Download Required Models

Download the following models and place them in the `assets/` directory:

- `assets/hubert/hubert_base.pt` - Hubert model for feature extraction
- `assets/pretrained/` - Pretrained models (v1)
- `assets/pretrained_v2/` - Pretrained models (v2, optional)
- `assets/uvr5_weights/` - UVR5 vocal separation models
- `assets/rmvpe/rmvpe.pt` - RMVPE pitch extraction model

You can download these from the [Hugging Face repository](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/).

### 4. Install FFmpeg

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt install ffmpeg
```

**Windows:**
Download `ffmpeg.exe` and `ffprobe.exe` and place them in the project root.

### 5. Configure Environment Variables

Copy the example environment file and configure it:

```bash
cp .env.example .env
```

Edit `.env` and set the following paths:

```env
weight_root=assets/weights
weight_uvr5_root=assets/uvr5_weights
index_root=logs
rmvpe_root=assets/rmvpe
```

### 6. Launch the WebUI

```bash
python infer-web.py
```

The WebUI will be available at `http://localhost:7865` (or the port specified in your configuration).

## 📖 Usage

### Voice Conversion

1. **Load a Model**: Select a trained model from the dropdown
2. **Upload Audio**: Upload the audio file you want to convert
3. **Configure Settings**:
   - **F0 Method**: Choose pitch extraction method (pm, harvest, crepe, rmvpe)
   - **Pitch Shift**: Adjust the pitch key (semitones)
   - **Index Rate**: Set the retrieval index strength (0.0-1.0)
   - **Filter Radius**: Set the median filter radius
4. **Convert**: Click "Convert" and wait for processing
5. **Download**: Download the converted audio

### Training a Model

1. **Prepare Dataset**: Place your audio files in a dataset folder (e.g., `logs/your_model_name/`)
2. **Preprocess**: Go to the Training tab and click "Preprocess"
3. **Extract Features**: Extract F0 and features
4. **Train**: Configure training parameters and start training
5. **Generate Index**: After training, generate the feature index for better quality

### Command Line Interface

You can also use the command-line tools:

**Single file conversion:**
```bash
python tools/infer_cli.py \
    --model_name "your_model" \
    --input_path "input.wav" \
    --output_path "output.wav" \
    --f0method pm \
    --index_rate 0.5
```

**Train index:**
```bash
python tools/train_index_cli.py \
    --model_name "your_model" \
    --version v2
```

## 🛠️ Project Structure

```
RVC/
├── assets/              # Model assets and weights
│   ├── hubert/         # Hubert model
│   ├── pretrained/     # Pretrained models (v1)
│   ├── pretrained_v2/  # Pretrained models (v2)
│   ├── rmvpe/          # RMVPE model
│   ├── uvr5_weights/   # UVR5 models
│   └── weights/        # Trained model weights
├── configs/            # Configuration files
├── infer/              # Inference modules
├── logs/               # Training logs and data
├── opt/                # Output files
├── tools/              # Command-line tools
├── infer-web.py        # Main WebUI script
├── .env.example        # Environment variables template
└── README.md           # This file
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file based on `.env.example`:

- `weight_root`: Path to trained model weights directory
- `weight_uvr5_root`: Path to UVR5 model weights
- `index_root`: Path to feature index directory
- `outside_index_root`: External index directory (optional)
- `rmvpe_root`: Path to RMVPE model directory

### macOS Specific Notes

This project includes special handling for macOS to prevent segmentation faults:
- MPS (Metal Performance Shaders) is disabled by default
- CPU mode is forced for stability
- Multiprocessing is configured for macOS compatibility

## 📝 Training Tips

- **Dataset Quality**: Use clean, high-quality audio (16kHz or higher, mono)
- **Dataset Size**: Minimum 10 minutes, recommended 30+ minutes
- **Audio Format**: WAV format is recommended
- **Training Epochs**: Start with 20-30 epochs, increase for better quality
- **Batch Size**: Adjust based on your GPU memory
- **Index Training**: Always train the feature index after model training for best results

## 🐛 Troubleshooting

### Common Issues

**Segmentation Fault on macOS:**
- The project is configured to use CPU mode on macOS by default
- If issues persist, ensure all dependencies are correctly installed

**Model Not Found:**
- Check that your `.env` file has correct paths
- Ensure model files are in the specified directories

**Out of Memory:**
- Reduce batch size during training
- Use lower quality settings during inference
- Close other applications using GPU memory

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

This project is based on the [RVC-Project](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) and uses the following technologies:

- [ContentVec](https://github.com/auspicious3000/contentvec/) - Feature extraction
- [VITS](https://github.com/jaywalnut310/vits) - Voice synthesis
- [HIFIGAN](https://github.com/jik876/hifi-gan) - Vocoder
- [Gradio](https://github.com/gradio-app/gradio) - Web interface
- [RMVPE](https://github.com/Dream-High/RMVPE) - Pitch extraction

## 📞 Support

For issues, questions, or contributions, please open an issue on GitHub.

---

**Note**: This is a modified version optimized for macOS compatibility and includes additional command-line tools for easier workflow automation.
