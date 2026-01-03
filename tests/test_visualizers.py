import pytest
import numpy as np

VISUALIZATION_TYPES = [
    "kaleidoscope",
    "waveform",
    "spectrum",
    "circular_spectrum",
    "dancing_particles",
    "bars_3d",
    "pulse",
    "equalizer",
    "spectrogram",
    "MD_spectrogram",
]


@pytest.mark.parametrize("viz_type", VISUALIZATION_TYPES)
def test_visualizations_output_shape(
    visualizer, mono_audio_samples, sample_rate, viz_type
):
    # Test at t=0.5s
    t = 0.5
    frame = visualizer.create_frame(t, mono_audio_samples, sample_rate, viz_type)

    assert frame.shape == (visualizer.height, visualizer.width, 3)
    assert frame.dtype == np.uint8


@pytest.mark.parametrize("viz_type", VISUALIZATION_TYPES)
def test_visualization_out_of_bounds(visualizer, audio_samples, sample_rate, viz_type):
    # Test time after audio ends
    t = 2.0
    frame = visualizer.create_frame(t, audio_samples, sample_rate, viz_type)
    # Should return black frame
    assert np.all(frame == 0)


def test_create_frame_invalid_type(visualizer, audio_samples, sample_rate):
    with pytest.raises(ValueError):
        visualizer.create_frame(0, audio_samples, sample_rate, "non_existent_viz")


@pytest.mark.parametrize("viz_type", VISUALIZATION_TYPES)
def test_all_visualizations_exception_handling(visualizer, viz_type):
    # Force exception by mocking the specific method in the dictionary
    from unittest.mock import MagicMock

    mock_func = MagicMock(side_effect=Exception("Forced Fail"))
    visualizer.visualization_functions[viz_type] = mock_func

    # Should catch exception and return black frame
    frame = visualizer.create_frame(0, np.zeros(100), 44100, viz_type)
    assert np.all(frame == 0)
    assert frame.shape == (visualizer.height, visualizer.width, 3)


def test_md_spectrogram_stereo_split(visualizer, audio_samples, sample_rate):
    # Explicitly test MD_spectrogram with stereo data to ensure split logic works
    visualizer.reset_buffers()
    frame = visualizer.MD_spectrogram_visualization(0.5, audio_samples, sample_rate)
    assert frame.shape == (visualizer.height, visualizer.width, 3)


def test_visualizer_inner_exceptions(
    visualizer, mono_audio_samples, sample_rate, mocker
):
    # Mock cv2 functions to raise exception
    # This ensures we hit the 'except' block INSIDE the visualization function
    mock_cv2 = mocker.patch("mdkali.cv2")
    mock_cv2.circle.side_effect = Exception("CV2 Fail")
    mock_cv2.line.side_effect = Exception("CV2 Fail")
    mock_cv2.rectangle.side_effect = Exception("CV2 Fail")
    mock_cv2.fillPoly.side_effect = Exception("CV2 Fail")
    mock_cv2.resize.side_effect = Exception("CV2 Fail")
    mock_cv2.GaussianBlur.side_effect = Exception("CV2 Fail")

    for viz_type in VISUALIZATION_TYPES:
        # We need to make sure we don't trigger the create_frame exception handler,
        # but rather the one inside the viz function.
        # But wait, create_frame calls the viz function inside a try...except too.
        # So if the viz function catches the exception, it returns zero frame.
        # If it DOESN'T catch it, create_frame catches it.
        # The viz functions DO have try...except blocks that return zero frame.
        # So if we raise inside, the viz function catches it.

        frame = visualizer.create_frame(0.5, mono_audio_samples, sample_rate, viz_type)
        assert np.all(frame == 0)
