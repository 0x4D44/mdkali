import pytest
from mdkali import main


def test_main_no_files(mocker):
    mocker.patch("sys.argv", ["mdkali.py", "*.nonexistent"])
    mocker.patch("glob.glob", return_value=[])

    mock_logger = mocker.patch("mdkali.logger")

    with pytest.raises(SystemExit):
        main()

    mock_logger.error.assert_called_with("No input files found")


def test_main_single_file(mocker):
    mocker.patch("sys.argv", ["mdkali.py", "test.mp3", "-o", "out.mp4"])
    mocker.patch("glob.glob", return_value=["test.mp3"])

    mock_viz_cls = mocker.patch("mdkali.AudioVisualizer")
    mock_viz = mock_viz_cls.return_value

    main()

    mock_viz.create_video.assert_called_with("test.mp3", "out.mp4", "MD_spectrogram")


def test_main_multiple_files(mocker):
    mocker.patch("sys.argv", ["mdkali.py", "*.mp3", "--visualization", "waveform"])
    mocker.patch("glob.glob", return_value=["1.mp3", "2.mp3"])

    mock_viz_cls = mocker.patch("mdkali.AudioVisualizer")
    mock_viz = mock_viz_cls.return_value

    main()

    assert mock_viz.create_video.call_count == 2
    mock_viz.create_video.assert_any_call("1.mp3", "1.mp4", "waveform")
    mock_viz.create_video.assert_any_call("2.mp3", "2.mp4", "waveform")


def test_main_extract_art(mocker):
    mocker.patch("sys.argv", ["mdkali.py", "test.mp3", "--extract-art"])
    mocker.patch("glob.glob", return_value=["test.mp3"])

    mock_extract = mocker.patch("mdkali.extract_album_art")
    mocker.patch("mdkali.AudioVisualizer")

    main()

    mock_extract.assert_called_with("test.mp3")


def test_main_verbose(mocker):
    mocker.patch("sys.argv", ["mdkali.py", "test.mp3", "--verbose"])
    mocker.patch("glob.glob", return_value=["test.mp3"])
    mocker.patch("mdkali.AudioVisualizer")

    mock_logger = mocker.patch("mdkali.logger")

    main()

    mock_logger.setLevel.assert_called()


def test_main_processing_error(mocker):
    mocker.patch("sys.argv", ["mdkali.py", "test.mp3"])
    mocker.patch("glob.glob", return_value=["test.mp3"])

    mock_viz_cls = mocker.patch("mdkali.AudioVisualizer")
    mock_viz = mock_viz_cls.return_value
    mock_viz.create_video.side_effect = Exception("Fail")

    mock_logger = mocker.patch("mdkali.logger")

    # Should not raise, just log error
    main()

    mock_logger.error.assert_called()
