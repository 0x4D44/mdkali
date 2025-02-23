# mdkali - Audio Visualizer

A sophisticated Python-based audio visualization tool that converts audio files into stunning visual representations. This project offers multiple visualization styles and supports both MP3 and WAV audio formats.

## Features

- 10 unique visualization styles:
  - Multi-Display Spectrogram (stereo channels)
  - Kaleidoscope
  - Waveform
  - Frequency Spectrum
  - Circular Spectrum
  - Dancing Particles
  - 3D Bars
  - Pulse Visualization
  - Equalizer
  - Spectrogram

- Additional features:
  - Album art extraction from MP3 files
  - Support for batch processing multiple files
  - Customizable output resolution and frame rate
  - Progress bar for rendering status
  - Comprehensive error handling and logging

## Requirements

```
numpy
opencv-python
moviepy
scipy
tqdm
mutagen
Pillow
```

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd audio-visualizer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python audio_visualizer.py input.mp3
```

This will create an MP4 file with the default Multi-Display Spectrogram visualization.

### Advanced Usage

```bash
python audio_visualizer.py --visualization kaleidoscope --width 1920 --height 1080 --fps 30 input.mp3 --output output.mp4
```

### Multiple Files

```bash
python audio_visualizer.py *.mp3 --visualization equalizer
```

### Command Line Arguments

- `input_files`: Input MP3 file(s) (supports wildcards)
- `--output`, `-o`: Output MP4 file (ignored for multiple inputs)
- `--visualization`, `-v`: Visualization type (default: MD_spectrogram)
- `--width`, `-w`: Video width (default: 1920)
- `--height`, `-H`: Video height (default: 1080)
- `--fps`, `-f`: Frames per second (default: 30)
- `--verbose`: Enable verbose logging
- `--extract-art`, `-e`: Extract album art from MP3 files

### Available Visualization Types

1. `MD_spectrogram`: Multi-display spectrogram showing stereo channels
2. `kaleidoscope`: Dynamic kaleidoscope pattern reacting to audio
3. `waveform`: Traditional audio waveform visualization
4. `spectrum`: Frequency spectrum analyzer
5. `circular_spectrum`: Circular frequency spectrum display
6. `dancing_particles`: Particle system reacting to audio
7. `bars_3d`: 3D frequency bars with perspective
8. `pulse`: Pulsing circular visualization
9. `equalizer`: Classic equalizer bar display
10. `spectrogram`: Scrolling spectrogram up to 12kHz

## Technical Details

### Audio Processing

- Supports both mono and stereo audio
- Automatic conversion of MP3 to WAV for processing
- FFT-based frequency analysis
- Hanning window applied to reduce spectral leakage
- Sample rate of 44.1kHz
- Maximum frequency display of 12kHz for spectrograms

### Video Output

- H.264 video codec
- AAC audio codec
- Configurable resolution and frame rate
- Multi-threaded rendering
- Progress tracking with tqdm

## Error Handling

The program includes comprehensive error handling for:
- File not found errors
- Invalid audio files
- Memory issues
- FFmpeg processing errors
- Invalid visualization parameters

Errors are logged with appropriate detail level based on verbosity settings.

## Examples

### Create a kaleidoscope visualization:
```bash
python audio_visualizer.py --visualization kaleidoscope input.mp3
```

### Process multiple files with album art extraction:
```bash
python audio_visualizer.py --extract-art --visualization equalizer *.mp3
```

### Create high-resolution output:
```bash
python audio_visualizer.py --width 3840 --height 2160 --fps 60 input.mp3
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

[Insert your chosen license here]

## Acknowledgments

This project uses several open-source libraries:
- NumPy for numerical operations
- OpenCV for image processing
- MoviePy for video creation
- SciPy for audio processing
- Mutagen for MP3 metadata handling
- PIL for image processing
