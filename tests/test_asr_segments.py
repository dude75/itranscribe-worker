from types import SimpleNamespace

from app.engines.base import Word, words_from_asr_segments


def test_whisper_style_words_keep_segment_id() -> None:
    segments = [
        SimpleNamespace(
            words=[
                SimpleNamespace(start=0.0, end=0.2, word=" hello"),
                SimpleNamespace(start=0.2, end=0.4, word="world"),
            ],
            start=0.0,
            end=0.4,
            text="hello world",
        ),
        SimpleNamespace(
            words=[SimpleNamespace(start=1.0, end=1.2, word="bye")],
            start=1.0,
            end=1.2,
            text="bye",
        ),
    ]
    words = words_from_asr_segments(segments)
    assert [w.text for w in words] == ["hello", "world", "bye"]
    assert [w.segment_id for w in words] == [0, 0, 1]
    assert words[0].start == 0.0
    assert words[2].end == 1.2


def test_gigaam_style_words_keep_segment_id() -> None:
    segments = [
        SimpleNamespace(
            words=[
                SimpleNamespace(start=0.0, end=0.3, text="привет"),
                SimpleNamespace(start=0.3, end=0.6, text="мир"),
            ],
            start=0.0,
            end=0.6,
            text="привет мир",
        ),
        SimpleNamespace(
            words=[SimpleNamespace(start=2.0, end=2.4, text="пока")],
            start=2.0,
            end=2.4,
            text="пока",
        ),
    ]
    words = words_from_asr_segments(segments)
    assert [w.text for w in words] == ["привет", "мир", "пока"]
    assert [w.segment_id for w in words] == [0, 0, 1]


def test_fallback_to_segment_text_when_no_words() -> None:
    segments = [
        SimpleNamespace(words=None, start=0.0, end=1.0, text=" whole phrase "),
    ]
    assert words_from_asr_segments(segments) == [
        Word(start=0.0, end=1.0, text="whole phrase", segment_id=0)
    ]


def test_skips_blank_words() -> None:
    segments = [
        SimpleNamespace(
            words=[
                SimpleNamespace(start=0.0, end=0.1, word="  "),
                SimpleNamespace(start=0.1, end=0.3, word="ok"),
            ],
            start=0.0,
            end=0.3,
            text="ok",
        )
    ]
    words = words_from_asr_segments(segments)
    assert [w.text for w in words] == ["ok"]
    assert words[0].segment_id == 0
