import pytest
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mdkali import AudioVisualizer


@pytest.fixture
def visualizer():
    return AudioVisualizer(width=100, height=100, fps=30)


@pytest.fixture
def sample_rate():
    return 44100


@pytest.fixture
def audio_samples(sample_rate):
    # 1 second of stereo audio
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    left = 0.5 * np.sin(2 * np.pi * 440 * t)
    right = 0.5 * np.sin(2 * np.pi * 880 * t)
    return np.column_stack((left, right)).astype(np.float32)


@pytest.fixture
def mono_audio_samples(sample_rate):
    # 1 second of mono audio
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    return (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
