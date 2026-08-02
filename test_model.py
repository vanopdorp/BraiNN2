import torch
import pytest
from model import MODEL_DIM, EmbodiedSpeechModel, SPARSITY

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") 


@pytest.fixture
def model():
    DIM = MODEL_DIM
    K = max(1, int(DIM * SPARSITY))
    return EmbodiedSpeechModel(DIM, K).to(DEVICE)

def test_audio_frontend_mel_shape(model):
    wav = torch.randn(1, 16000, device=DEVICE)
    mel, vad = model.audio_frontend(wav)
    assert mel.ndim == 4
    assert mel.shape[1] == 1
    assert vad.shape == torch.Size([1])

def test_audio_encoder_output(model):
    wav = torch.randn(1, 16000, device=DEVICE)
    mel, vad = model.audio_frontend(wav)
    vals, idx, event = model.audio_encoder(mel, vad)
    assert vals.ndim == 2
    assert idx.ndim == 1
    assert isinstance(event, bool)

def test_vision_encoder_output(model):
    frame = torch.randn(1, 3, 128, 128, device=DEVICE)
    vals, idx, event = model.vision(frame)
    assert vals.ndim == 2
    assert idx.ndim == 1
    assert isinstance(event, bool)

def test_fusion_layer(model):
    frame = torch.randn(1, 3, 128, 128, device=DEVICE)
    wav = torch.randn(1, 16000, device=DEVICE)
    v_vals, v_idx, _ = model.vision(frame)
    mel, vad = model.audio_frontend(wav)
    a_vals, a_idx, _ = model.audio_encoder(mel, vad)

    B, dim = v_vals.size()
    dense_v = torch.zeros(B, dim, device=DEVICE, dtype=torch.float16)
    dense_a = torch.zeros(B, dim, device=DEVICE, dtype=torch.float16)
    dense_v[:, v_idx] = v_vals
    dense_a[:, a_idx] = a_vals

    fused = dense_v + dense_a
    full_idx = torch.arange(dim, device=DEVICE)
    out_vals, out_idx = model.fusion_layer(fused, full_idx)

    assert out_vals.ndim == 2
    assert out_idx.ndim == 1

def test_full_brain_forward(model):
    frame = torch.randn(1, 3, 128, 128, device=DEVICE)
    wav = torch.randn(1, 16000, device=DEVICE)
    out, vad = model.step(frame, wav)

    assert "meaning_vals" in out
    assert "form_vals" in out
    assert "recon" in out

    assert out["meaning_vals"].ndim == 2
    assert out["form_vals"].ndim == 2
    assert out["recon"].ndim == 2

def test_vad_boolean(model):
    wav = torch.randn(1, 16000, device=DEVICE)
    _, vad = model.step(torch.randn(1, 3, 128, 128, device=DEVICE), wav)
    assert isinstance(bool(vad.item()), bool)
