from __future__ import annotations

import pytest

from app.config import get_settings
from app.engines.diarization.nemo import (
    NemoSortformerDiarizer,
    NemoUnavailable,
    _import_sortformer,
    segments_from_sortformer,
)
from app.pipeline import checkpoints_for
from app.schemas import AsrModel, DiarizationModel


def test_segments_from_sortformer_format() -> None:
    nested = [["0.50 1.70 speaker_0", "2.00 2.80 speaker_1"]]
    segments = segments_from_sortformer(nested)
    assert len(segments) == 2
    assert segments[0].start == pytest.approx(0.5)
    assert segments[0].end == pytest.approx(1.7)
    assert segments[0].speaker == "speaker_0"
    assert segments[1].speaker == "speaker_1"

    triples = [[0.5, 1.7, "speaker_0"], [2.0, 2.8, "speaker_1"]]
    assert len(segments_from_sortformer(triples)) == 2


def test_nemo_checkpoint_name() -> None:
    settings = get_settings()
    _asr, diar = checkpoints_for(settings, AsrModel.whisper, DiarizationModel.nemo)
    assert diar == settings.NEMO_MODEL
    assert "sortformer" in diar


def test_no_diarization_checkpoint() -> None:
    settings = get_settings()
    asr, diar = checkpoints_for(settings, AsrModel.whisper, None)
    assert asr == settings.WHISPER_MODEL
    assert diar is None


def test_sortformer_unavailable_app_stays_importable() -> None:
    import app

    assert app is not None
    try:
        _import_sortformer()
    except NemoUnavailable:
        with pytest.raises(NemoUnavailable):
            NemoSortformerDiarizer("missing", "data/models")
        return


@pytest.mark.ml
def test_sortformer_engine_loads() -> None:
    pytest.importorskip("nemo")
    settings = get_settings()
    try:
        _import_sortformer()
    except NemoUnavailable:
        pytest.skip("SortformerEncLabelModel is not in this NeMo build")
    engine = NemoSortformerDiarizer(
        settings.NEMO_MODEL,
        settings.MODELS_DIR,
        hf_token=settings.HF_TOKEN,
    )
    assert engine is not None
