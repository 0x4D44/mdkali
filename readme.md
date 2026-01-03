# mdkali - Audio Visualizer

A sophisticated Python-based audio visualization tool that converts audio files (MP3/WAV) into stunning video representations. This project leverages signal processing (FFT) to drive 10 unique visualization styles, from classic waveforms to complex kaleidoscopes and 3D bars.

## Features

- **10 Unique Visualization Styles:**
  - `MD_spectrogram`: Multi-display with stereo waveforms and scrolling spectrograms (Default).
  - `kaleidoscope`: Dynamic kaleidoscope pattern reacting to audio amplitude.
  - `waveform`: Traditional audio waveform visualization.
  - `spectrum`: Frequency spectrum analyzer bars.
  - `circular_spectrum`: Circular frequency spectrum display.
  - `dancing_particles`: Particle system where energy drives movement.
  - `bars_3d`: 3D frequency bars with perspective.
  - `pulse`: Pulsing circular glow based on beat detection.
  - `equalizer`: Classic 32-band equalizer.
  - `spectrogram`: Scrolling frequency heatmap (up to 12kHz).

- **Additional Capabilities:**
  - **Batch Processing:** Process multiple files at once using wildcards (`*.mp3`).
  - **Album Art:** Extract embedded album art from MP3s (`--extract-art`).
  - **High Performance:** Uses `scipy` for FFT and `opencv` for rendering.
  - **Customizable:** Adjust resolution, frame rate, and output paths.

## Requirements

- **Python 3.8+**
- **FFmpeg:** Required for audio processing and video encoding.
  - *Windows:* Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH.
  - *Linux:* `sudo apt install ffmpeg`
  - *macOS:* `brew install ffmpeg`

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/mdkali.git
    cd mdkali
    ```

2.  **Create a virtual environment (Recommended):**
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # Linux/macOS
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The primary entry point is `mdkali.py`.

### Basic Usage
Create a video with the default `MD_spectrogram` visualization:
```bash
python mdkali.py input.mp3
```
This generates `input.mp4` in the same directory.

### Advanced Usage
Customize visualization style, resolution, and frame rate:
```bash
python mdkali.py input.mp3 --visualization kaleidoscope --width 1920 --height 1080 --fps 60 --output my_video.mp4
```

### Batch Processing
Process all MP3 files in the current directory:
```bash
python mdkali.py *.mp3 --visualization equalizer
```

### Command Line Arguments

| Argument | Description | Default |
| :--- | :--- | :--- |
| `input_files` | Input MP3/WAV file(s) (supports wildcards like `*.mp3`) | Required |
| `--output`, `-o` | Output video filename (ignored for batch processing) | Derived from input |
| `--visualization`, `-v` | Visualization style (see list above) | `MD_spectrogram` |
| `--width`, `-w` | Video width | `1920` |
| `--height`, `-H` | Video height | `1080` |
| `--fps`, `-f` | Frames per second | `30` |
| `--extract-art`, `-e` | Extract embedded album art as image files | `False` |
| `--verbose` | Enable detailed logging | `False` |

## Development

### Running Tests
The project uses `pytest` for testing. Code coverage is currently **99%**.

1.  Install test dependencies:
    ```bash
    pip install pytest pytest-cov pytest-mock
    ```

2.  Run tests:
    ```bash
    python -m pytest
    ```

3.  Run tests with coverage report:
    ```bash
    python -m pytest --cov=mdkali --cov-report term-missing tests
    ```

### Linting and Formatting
The project enforces code style using `ruff`.

1.  Install ruff:
    ```bash
    pip install ruff
    ```

2.  Check for linting errors:
    ```bash
    python -m ruff check .
    ```

3.  Format code:
    ```bash
    python -m ruff format .
    ```

## License

This project is licensed under the MIT License - see the LICENSE file for details.