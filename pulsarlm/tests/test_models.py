import pytest

from pulsarlm.models import MODELS, HuggingFaceRepository, hugging_face_repository


def test_a_hugging_face_url_gives_its_repository_and_revision():
    assert hugging_face_repository(
        "https://huggingface.co/Qwen/Qwen3-8B/tree/9213176"
    ) == HuggingFaceRepository("Qwen/Qwen3-8B", "9213176")


@pytest.mark.parametrize(
    "url",
    [
        "https://huggingface.co/Qwen/Qwen3-8B",
        "https://huggingface.co/Qwen/Qwen3-8B/tree/",
        "https://example.com/Qwen/Qwen3-8B/tree/main",
    ],
)
def test_a_url_without_a_hugging_face_repository_and_revision_is_rejected(url):
    with pytest.raises(ValueError, match="not a Hugging Face repository"):
        hugging_face_repository(url)


def test_every_model_url_names_a_repository_and_revision():
    for source in MODELS.values():
        hugging_face_repository(source.url)
