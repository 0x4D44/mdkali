from unittest.mock import MagicMock
from mdkali import extract_album_art


def test_extract_album_art_success(mocker):
    # Mock MP3
    mock_mp3 = mocker.patch("mdkali.MP3")
    mock_audio = MagicMock()
    mock_mp3.return_value = mock_audio

    # Mock Tag
    mock_tag = MagicMock()
    mock_tag.FrameID = "APIC"
    mock_tag.data = b"imagedata"
    mock_audio.tags = {"APIC": mock_tag}

    # Mock Image
    mock_image_cls = mocker.patch("mdkali.Image")
    mock_image = MagicMock()
    mock_image.format = "JPEG"
    mock_image_cls.open.return_value = mock_image

    # Mock IO
    mocker.patch("mdkali.io.BytesIO")

    result = extract_album_art("test.mp3")

    assert result is True
    mock_image.save.assert_called_once_with("test.jpeg")


def test_extract_album_art_no_tags(mocker):
    mock_mp3 = mocker.patch("mdkali.MP3")
    mock_audio = MagicMock()
    mock_audio.tags = {}
    mock_mp3.return_value = mock_audio

    result = extract_album_art("test.mp3")
    assert result is False


def test_extract_album_art_exception(mocker):
    mocker.patch("mdkali.MP3", side_effect=Exception("Boom"))
    result = extract_album_art("test.mp3")
    assert result is False
