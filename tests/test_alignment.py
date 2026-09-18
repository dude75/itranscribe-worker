from app.alignment import SPEAKER_UNKNOWN, align, utterances_from_words
from app.engines.base import DiarizationSegment, Word


def test_word_fully_inside_segment() -> None:
    words = [Word(start=1.0, end=1.5, text="привет")]
    segments = [DiarizationSegment(start=0.0, end=2.0, speaker="A")]
    result = align(words, segments)
    assert len(result) == 1
    assert result[0].speaker == "speaker_0"
    assert result[0].text == "привет"
    assert result[0].start == 1.0
    assert result[0].end == 1.5


def test_boundary_uses_word_center() -> None:
    words = [Word(start=0.9, end=1.3, text="на")]
    segments = [
        DiarizationSegment(start=0.0, end=1.0, speaker="A"),
        DiarizationSegment(start=1.0, end=2.0, speaker="B"),
    ]
    result = align(words, segments)
    assert len(result) == 1
    assert result[0].speaker == "speaker_0"
    assert result[0].text == "на"


def test_unknown_when_no_segment() -> None:
    words = [Word(start=5.0, end=5.2, text="э")]
    segments = [DiarizationSegment(start=0.0, end=1.0, speaker="A")]
    result = align(words, segments)
    assert result[0].speaker == SPEAKER_UNKNOWN


def test_speaker_n_by_first_appearance() -> None:
    words = [
        Word(start=0.0, end=0.2, text="один"),
        Word(start=1.0, end=1.2, text="два"),
        Word(start=2.0, end=2.2, text="три"),
    ]
    segments = [
        DiarizationSegment(start=1.0, end=1.5, speaker="SPEAKER_01"),
        DiarizationSegment(start=0.0, end=0.5, speaker="SPEAKER_00"),
        DiarizationSegment(start=2.0, end=2.5, speaker="SPEAKER_00"),
    ]
    result = align(words, segments)
    assert [u.speaker for u in result] == ["speaker_0", "speaker_1", "speaker_0"]
    assert result[0].text == "один"
    assert result[1].text == "два"
    assert result[2].text == "три"


def test_merge_consecutive_same_speaker() -> None:
    words = [
        Word(start=0.0, end=0.2, text="раз"),
        Word(start=0.2, end=0.4, text="два"),
        Word(start=1.0, end=1.2, text="три"),
    ]
    segments = [
        DiarizationSegment(start=0.0, end=0.5, speaker="A"),
        DiarizationSegment(start=1.0, end=1.5, speaker="B"),
    ]
    result = align(words, segments)
    assert len(result) == 2
    assert result[0].speaker == "speaker_0"
    assert result[0].text == "раз два"
    assert result[0].start == 0.0
    assert result[0].end == 0.4
    assert result[1].speaker == "speaker_1"
    assert result[1].text == "три"


def test_utterances_from_words_no_speaker() -> None:
    words = [
        Word(start=0.0, end=0.2, text="привет"),
        Word(start=0.2, end=0.4, text="мир"),
    ]
    result = utterances_from_words(words)
    assert len(result) == 1
    assert result[0].speaker is None
    assert result[0].text == "привет мир"
    assert result[0].start == 0.0
    assert result[0].end == 0.4


def test_utterances_from_words_split_on_segment_id() -> None:
    words = [
        Word(start=0.0, end=0.2, text="привет", segment_id=0),
        Word(start=0.2, end=0.4, text="мир", segment_id=0),
        Word(start=1.0, end=1.2, text="пока", segment_id=1),
    ]
    result = utterances_from_words(words)
    assert [u.text for u in result] == ["привет мир", "пока"]
    assert all(u.speaker is None for u in result)
    assert result[0].start == 0.0
    assert result[0].end == 0.4
    assert result[1].start == 1.0
    assert result[1].end == 1.2


def test_align_splits_same_speaker_on_segment_id() -> None:
    words = [
        Word(start=0.0, end=0.2, text="раз", segment_id=0),
        Word(start=0.2, end=0.4, text="два", segment_id=0),
        Word(start=1.0, end=1.2, text="три", segment_id=1),
    ]
    segments = [DiarizationSegment(start=0.0, end=2.0, speaker="A")]
    result = align(words, segments)
    assert len(result) == 2
    assert result[0].speaker == "speaker_0"
    assert result[1].speaker == "speaker_0"
    assert result[0].text == "раз два"
    assert result[1].text == "три"


def test_align_speaker_change_inside_same_segment() -> None:
    words = [
        Word(start=0.0, end=0.2, text="я", segment_id=0),
        Word(start=0.5, end=0.7, text="ты", segment_id=0),
    ]
    segments = [
        DiarizationSegment(start=0.0, end=0.3, speaker="A"),
        DiarizationSegment(start=0.4, end=1.0, speaker="B"),
    ]
    result = align(words, segments)
    assert [u.speaker for u in result] == ["speaker_0", "speaker_1"]
    assert [u.text for u in result] == ["я", "ты"]


def test_utterances_from_words_empty() -> None:
    assert utterances_from_words([]) == []
