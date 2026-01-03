import pytest
import numpy as np
from mdkali import AudioVisualizer


def test_init_valid():
    viz = AudioVisualizer(1920, 1080, 30)
    assert viz.width == 1920
    assert viz.height == 1080
    assert viz.fps == 30
    assert len(viz.particles) == 100


def test_init_invalid():
    with pytest.raises(ValueError):
        AudioVisualizer(-1, 100, 30)
    with pytest.raises(ValueError):
        AudioVisualizer(100, -1, 30)
    with pytest.raises(ValueError):
        AudioVisualizer(100, 100, 0)


def test_reset_buffers(visualizer):
    # Manually modify buffer
    visualizer.spec_buffer.fill(255)
    visualizer.md_spec_buffers = {"left": "dummy", "right": "dummy"}

    visualizer.reset_buffers()

    assert np.all(visualizer.spec_buffer == 0)
    assert visualizer.md_spec_buffers["left"].shape[0] > 0
    assert visualizer.md_spec_buffers["left"].dtype == np.uint8


def test_init_particles(visualizer):
    visualizer.particles = []
    visualizer.init_particles(50)
    assert len(visualizer.particles) == 50
    p = visualizer.particles[0]
    assert "x" in p and "y" in p and "vx" in p and "vy" in p
