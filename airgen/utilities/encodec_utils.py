import os
import sys

import torch
from .utils import get_device
from demucs.audio import convert_audio

sys.path.insert(0, os.path.join(sys.path[0], "../../.."))
from ..audiocraft.models.loaders import load_compression_model, load_lm_model
from ..audiocraft.data.audio import audio_write

cache_dir = os.environ.get('MUSICGEN_ROOT', None)
name = None
device = get_device()
compression_model = load_compression_model("large", device=device, cache_dir=cache_dir)
compression_model.eval()
lm = load_lm_model("large", device=device, cache_dir=cache_dir)
lm.eval()
sample_rate = 32000


def extract_rvq(x, sr=32000):
    if not sr == sample_rate:
        x = convert_audio(x, sr, sample_rate, 1)
    with torch.no_grad():
        seq, _ = compression_model.encode(x.to(device))
    return seq.squeeze(0)


def extract_musicgen_emb(seq):
    seq = seq[None, ...]
    with torch.no_grad():
        emb = sum([lm.emb[i](seq[:, i]) for i in range(4)])
    return emb.squeeze(0)


def save_rvq(output_list, tokens):
    with torch.no_grad():
        gen_audio = compression_model.decode(tokens, None)
        for i in range(gen_audio.size(0)):
            pred_wav = gen_audio[i].cpu()
            audio_write(output_list[i], pred_wav.cpu(), sample_rate, strategy="loudness", loudness_compressor=True)
