# mdkali - Audio Visualizer

## Project Overview

**mdkali** is a Python-based CLI tool designed to convert audio files (MP3/WAV) into visually engaging MP4 videos. It leverages signal processing (FFT) to drive various visualization styles, ranging from classic waveforms and equalizers to complex kaleidoscopes and 3D bars.

## Prerequisites

- **Python 3.x**
- **FFmpeg**: Required by `moviepy` and for audio conversion.

## Installation

1.  Install Python dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: `moviepy` is pinned to version 1.0.3.*

## Usage

The primary entry point is `mdkali.py`.

### Basic Command
Convert an MP3 to MP4 using the default visualization (`MD_spectrogram`):
```bash
python mdkali.py input.mp3
```

### Advanced Options
Customize visualization style, resolution, and frame rate:
```bash
python mdkali.py input.mp3 --visualization kaleidoscope --width 1920 --height 1080 --fps 30 --output my_video.mp4
```

### Batch Processing
Process all MP3s in the current directory:
```bash
python mdkali.py *.mp3 --visualization equalizer
```

### Available Visualizations
- `MD_spectrogram` (Default: Stereo waveforms + spectrograms)
- `kaleidoscope`
- `waveform`
- `spectrum`
- `circular_spectrum`
- `dancing_particles`
- `bars_3d`
- `pulse`
- `equalizer`
- `spectrogram`

### Key Flags
- `--extract-art`, `-e`: Extract embedded album art from MP3s.
- `--verbose`: Enable detailed logging.

## Code Architecture

The core logic resides in `mdkali.py`:

*   **`AudioVisualizer` Class**: Manages the state, visualization buffers, and frame generation.
*   **Visualization Methods**: Each style has a corresponding method (e.g., `kaleidoscope_visualization`, `waveform_visualization`) that takes time `t`, `audio_samples`, and `sample_rate` to generate a single video frame using OpenCV (`cv2`).
*   **Audio Processing**:
    *   Uses `scipy.fftpack.fft` and `scipy.signal.spectrogram` for frequency analysis.
    *   `get_audio_data`: Converts MP3 to WAV (via ffmpeg) if necessary, reads samples, and normalizes them.
*   **Video Generation**:
    *   Uses `moviepy.video.VideoClip` to construct the video from the generated frames.
    *   Syncs the original audio track to the generated video.

## Development Notes

*   **Coordinate System**: Uses standard image coordinates (0,0 at top-left) via OpenCV.
*   **Buffers**: Some visualizations like `spectrogram` and `MD_spectrogram` maintain state buffers (`self.spec_buffer`, `self.md_spec_buffers`) that must be reset between files (`reset_buffers`).
*   **Performance**: Rendering is CPU-bound. `tqdm` is used to display progress.
