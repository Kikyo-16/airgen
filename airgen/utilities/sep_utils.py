import torch
from demucs.apply import apply_model
from demucs.audio import convert_audio
from demucs import pretrained

from .utils import get_device

device = get_device()
demucs_model = pretrained.get_model('htdemucs').to(device)
demucs_model.eval()

def separate(wav, sample_rate):
    #{'drums': 0, 'bass': 1, 'other': 2, 'vocal': 3}
    wav = convert_audio(wav, sample_rate, demucs_model.samplerate, demucs_model.audio_channels)
    with torch.no_grad():
        stems = apply_model(demucs_model, wav, device=device)
    wavs = {
        "bass": stems[:, 1].mean(1, keepdim=True),
        "drums": stems[:, 0].mean(1, keepdim=True),
        "other": stems[:, 2].mean(1, keepdim=True),
        "vocals": stems[:, 3].mean(1, keepdim=True)
    }
    for k in wavs:
        wavs[k] = convert_audio(wavs[k], demucs_model.samplerate, sample_rate, 1)
    return wavs

