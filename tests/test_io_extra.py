import pytest
import numpy as np


def test_get_audio_data_cleanup_error(visualizer, mocker):
    mocker.patch.object(visualizer, "convert_to_wav", return_value="temp.wav")
    mocker.patch("scipy.io.wavfile.read", return_value=(44100, np.array([0])))
    # Mock os.remove to fail
    mocker.patch("os.remove", side_effect=OSError("Access denied"))

    # Should not raise exception
    visualizer.get_audio_data("input.mp3")


def test_get_audio_data_exception(visualizer, mocker):
    mocker.patch.object(
        visualizer, "convert_to_wav", side_effect=Exception("Conversion failed")
    )

    with pytest.raises(Exception):
        visualizer.get_audio_data("input.mp3")
