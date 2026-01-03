import pytest
import numpy as np
import os
import subprocess
from unittest.mock import MagicMock, patch


def test_convert_to_wav_success(visualizer, mocker):
    mocker.patch("subprocess.run")
    mocker.patch("tempfile.mktemp", return_value="temp.wav")

    result = visualizer.convert_to_wav("input.mp3")

    assert result == "temp.wav"
    subprocess.run.assert_called_with(
        [
            "ffmpeg",
            "-i",
            "input.mp3",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "44100",
            "-ac",
            "1",
            "temp.wav",
        ],
        check=True,
        capture_output=True,
    )


def test_convert_to_wav_failure(visualizer, mocker):
    mocker.patch('tempfile.mktemp', return_value='temp.wav')
    mocker.patch(
        'subprocess.run',
        side_effect=subprocess.CalledProcessError(1, 'cmd', stderr=b'ffmpeg error'),
    )

    with pytest.raises(subprocess.CalledProcessError):
        visualizer.convert_to_wav('input.mp3')


def test_get_audio_data_wav(visualizer, mocker):
    # Test with direct WAV input
    mocker.patch(
        "scipy.io.wavfile.read",
        return_value=(44100, np.array([0, 100], dtype=np.int16)),
    )

    samples, rate = visualizer.get_audio_data("input.wav")

    assert rate == 44100
    assert len(samples) == 2
    # Check normalization (max is 100)
    assert np.max(samples) == 1.0


def test_get_audio_data_mp3(visualizer, mocker):
    # Test with MP3 (requires conversion)
    mocker.patch.object(visualizer, "convert_to_wav", return_value="temp.wav")
    mocker.patch(
        "scipy.io.wavfile.read", return_value=(44100, np.array([100], dtype=np.int16))
    )
    mocker.patch("os.remove")

    samples, rate = visualizer.get_audio_data("input.mp3")

    visualizer.convert_to_wav.assert_called_once_with("input.mp3")
    os.remove.assert_called_once_with("temp.wav")


def test_create_video_success(visualizer, mocker):
    mocker.patch("os.path.exists", return_value=True)
    mocker.patch("os.makedirs")

    # Mock audio data
    mocker.patch.object(
        visualizer, "get_audio_data", return_value=(np.zeros(44100), 44100)
    )

    # Mock tqdm
    mock_tqdm = mocker.patch("mdkali.tqdm")
    mock_pbar = mock_tqdm.return_value

    # Mock moviepy
    mock_video_clip_cls = mocker.patch('mdkali.VideoClip')
    mocker.patch('mdkali.AudioFileClip')

    mock_final_clip = MagicMock()
    mock_video_clip = mock_video_clip_cls.return_value
    mock_video_clip.set_audio.return_value = mock_final_clip

    visualizer.create_video("input.mp3", "output.mp4")

    # Retrieve the make_frame function passed to VideoClip
    args, kwargs = mock_video_clip_cls.call_args
    make_frame = args[0]

    # Call make_frame to trigger the inner logic
    # The inner logic calls create_frame, which we need to mock or let run.
    # Since we didn't mock create_frame, it will run (and might fail if logic is brittle or slow, but with zeros it should be fast/safe)
    # But create_frame relies on visualization functions.
    # Let's mock create_frame to avoid complex dependencies here, checking only the closure logic.

    with patch.object(
        visualizer, 'create_frame', return_value=np.zeros((100, 100, 3))
    ) as mock_create_frame:
        make_frame(0.1)
        assert mock_create_frame.called
        mock_pbar.update.assert_called()

    mock_final_clip.write_videofile.assert_called_once()


def test_create_video_file_not_found(visualizer):
    with pytest.raises(FileNotFoundError):
        visualizer.create_video('non_existent.mp3', 'out.mp4')
