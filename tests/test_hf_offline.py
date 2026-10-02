from __future__ import annotations

import os
import sys
import types

import pytest

from app.engines.hf_offline import (
    call_with_local_files_only,
    configure_models_dir,
    hf_hub_model_cached,
    huggingface_offline,
    huggingface_offline_enabled,
    looks_like_missing_cache,
)


def test_huggingface_offline_sets_and_restores_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    with huggingface_offline():
        assert os.environ["HF_HUB_OFFLINE"] == "1"
        assert huggingface_offline_enabled()
    assert "HF_HUB_OFFLINE" not in os.environ
    assert not huggingface_offline_enabled()


def test_huggingface_offline_nested_restores_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    with huggingface_offline():
        assert os.environ["HF_HUB_OFFLINE"] == "1"
        with huggingface_offline():
            assert os.environ["HF_HUB_OFFLINE"] == "1"
        assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert "HF_HUB_OFFLINE" not in os.environ


def test_huggingface_offline_syncs_constants_imported_inside(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delitem(sys.modules, "huggingface_hub.constants", raising=False)
    constants = types.ModuleType("huggingface_hub.constants")
    with huggingface_offline():
        constants.HF_HUB_OFFLINE = True
        monkeypatch.setitem(sys.modules, "huggingface_hub.constants", constants)
    assert constants.HF_HUB_OFFLINE is False


def test_looks_like_missing_cache() -> None:
    assert looks_like_missing_cache(
        RuntimeError("Cannot find an appropriate cached snapshot folder")
    )
    assert looks_like_missing_cache(RuntimeError("Model x was not found locally"))
    assert looks_like_missing_cache(
        OSError(
            "Can't load feature extractor for 'superb/wav2vec2-base-superb-er'. "
            "If you were trying to load it from 'https://huggingface.co/models'"
        )
    )
    assert not looks_like_missing_cache(RuntimeError("HF_TOKEN is empty. Accept the license"))
    assert not looks_like_missing_cache(RuntimeError("CUDA out of memory"))
    assert not looks_like_missing_cache(RuntimeError("weights missing"))


def test_configure_models_dir_sets_env_and_hub_constants(tmp_path, monkeypatch) -> None:
    monkeypatch.delitem(sys.modules, "huggingface_hub.constants", raising=False)
    import huggingface_hub.constants as constants

    target = tmp_path / "models"
    resolved = configure_models_dir(str(target))
    assert resolved == str(target.resolve())
    assert os.environ["HF_HUB_CACHE"] == resolved
    assert constants.HF_HUB_CACHE == resolved


def test_hf_hub_model_cached(tmp_path) -> None:
    models_dir = tmp_path / "models"
    assert not hf_hub_model_cached(str(models_dir), "nvidia/parakeet-tdt-0.6b-v3")
    repo = models_dir / "models--nvidia--parakeet-tdt-0.6b-v3" / "snapshots" / "abc"
    repo.mkdir(parents=True)
    assert hf_hub_model_cached(str(models_dir), "nvidia/parakeet-tdt-0.6b-v3")


def test_call_with_local_files_only_when_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)

    def supported(*, local_files_only: bool = False) -> bool:
        return local_files_only

    def unsupported() -> str:
        return "ok"

    assert call_with_local_files_only(supported) is False
    with huggingface_offline():
        assert call_with_local_files_only(supported) is True
        assert call_with_local_files_only(unsupported) == "ok"
