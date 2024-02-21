import os
import torch
import librosa
import argparse
from airgen.utilities import get_device, np2torch, mkdir, midi2wav
from airgen.models import AIRGen
from airgen.utilities.encodec_utils import extract_rvq, save_rvq
from airgen.utilities.sep_utils import separate

SOUNDFONT_PATH = "sf/MS_Basic.sf2"
SAMPLE_RATE = 32000
DUR = 15

device = get_device()


def load_data(audio_path: str, midi_path: str = None,
              drums_path: str = None, chord_path: str = None,
              beat_path: str = None, onset: int = 0,
              mode: str = 'fix_drums') -> (torch.Tensor, torch.Tensor):
    print("Start processing data...")
    onset = int(float(onset) * SAMPLE_RATE)

    audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    assert onset + DUR * SAMPLE_RATE <= len(audio)
    audio = audio[onset: int(onset + DUR * SAMPLE_RATE)]
    audio = np2torch(audio).to(device)[None, None, ...]
    wavs = separate(audio, sample_rate=SAMPLE_RATE)
    cond_audio = None
    cond_path = None

    if mode == "piano":
        assert midi_path is not None and os.path.exists(midi_path)

        from airgen.utilities.symbolic_utils import reduce_piano
        reduce_piano(midi_path, midi_path + ".tmp.piano.mid")
        cond_path = midi_path + ".tmp.piano.mid.wav"
        midi2wav(midi_path + ".tmp.piano.mid", cond_path, sf_path=SOUNDFONT_PATH, sample_rate=SAMPLE_RATE)

    elif mode == "chord":
        assert chord_path is not None and os.path.exists(chord_path)
        assert beat_path is not None and os.path.exists(beat_path)
        from airgen.utilities.symbolic_utils import chord_with_beat
        chord_with_beat(chord_path=chord_path,
                        beat_path=beat_path,
                        midi_path=chord_path + ".tmp.mid")
        cond_path = chord_path + ".mid.piano.tmp.wav"
        midi2wav(chord_path + ".tmp.mid", cond_path, sf_path=SOUNDFONT_PATH, sample_rate=SAMPLE_RATE)

    elif mode == "fix_drums":
        cond_audio = wavs["drums"]

    elif mode == "edit_drums":
        assert drums_path is not None and os.path.exists(drums_path)
        cond_path = drums_path

    if cond_audio is None:
        cond_audio, _ = librosa.load(cond_path, sr=SAMPLE_RATE, mono=True)
        assert onset + DUR * SAMPLE_RATE <= len(cond_audio)
        cond_audio = cond_audio[onset: int(onset + DUR * SAMPLE_RATE)]
        cond_audio = np2torch(cond_audio).to(device)[None, None, ...]
        if mode in ["piano", "chord"]:
            mean = torch.mean(wavs["other"])
            std = torch.std(wavs["other"])
            cond_audio = (cond_audio - torch.mean(cond_audio)) / torch.std(cond_audio)
            cond_audio = cond_audio * std + mean

    mix_rvq = extract_rvq(audio, sr=SAMPLE_RATE)
    cond_rvq = extract_rvq(cond_audio, sr=SAMPLE_RATE)
    return mix_rvq, cond_rvq


def get_mask_2(x):
    xlen = x.shape[-1] // 2
    mask = torch.ones_like(x)
    qlen = xlen // 2
    mask[qlen:qlen * 3] = 0
    return mask


def wrap_batch(mix_rvq, cond_rvq, n_samples):
    mask = get_mask_2(mix_rvq[0])
    idx = mask == 1
    cond_rvq[:, idx] = mix_rvq[:, idx]
    trk_id = torch.cat([mask, mask + 2, mask[-1:] + 2], -1)
    seq = cond_rvq[None, ...].repeat(n_samples, 1, 1).long()
    trk_id = trk_id[None, ...].repeat(n_samples, 1)
    print(seq.shape, trk_id.shape)

    prompt = [None] * n_samples
    batch = {
        "prompt": seq,
        "desc": prompt,
        "xlen": seq.shape[-1],
        "trk_id": trk_id.to(seq.device).long()
    }
    return batch


def save_preds(output_folder, batch, gen_tokens, n_samples, tag, with_prefix=False):
    pred = torch.cat([batch["prompt"], gen_tokens], -1) if with_prefix else gen_tokens
    output_path = [os.path.join(output_folder, f"{tag}_{j}") for j in range(n_samples)]
    save_rvq(output_list=output_path, tokens=pred)



def inference(args):
    output_folder = args.output_folder
    n_samples = args.num_samples
    mkdir(output_folder)
    mix_rvq, cond_rvq = load_data(audio_path=args.audio_path,
                                  midi_path=args.midi_path,
                                  chord_path=args.chord_path,
                                  drums_path=args.drums_path,
                                  beat_path=args.beat_path,
                                  mode=args.mode,
                                  onset=args.onset)
    print("Load model...")

    model = AIRGen(sec=30,
                   num_layers=48,
                   n_tasks=4,
                   k=50).to(device)
    model.load_weights(args.model_path)
    model.eval()
    print("Wrap batch...")
    batch = wrap_batch(mix_rvq, cond_rvq, n_samples=n_samples)
    print("Predicting...")
    with torch.no_grad():
        gen_tokens = model.generate(**batch)
    save_preds(output_folder,
               batch=batch,
               gen_tokens=gen_tokens,
               tag=args.mode,
               n_samples=n_samples)
    save_preds(output_folder,
               batch=batch,
               gen_tokens=batch["prompt"][:1],
               tag=args.mode + "_cond",
               n_samples=1)
    print("All done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_folder', type=str)
    parser.add_argument('-e', '--model_path', type=str)
    parser.add_argument('-m', '--mode', type=str)
    parser.add_argument('-a', '--audio_path', type=str)
    parser.add_argument('-p', '--midi_path', type=str, default='')
    parser.add_argument('-c', '--chord_path', type=str, default='')
    parser.add_argument('-b', '--beat_path', type=str, default='')
    parser.add_argument('-f', '--drums_path', type=str, default='')
    parser.add_argument('-s', '--onset', type=int, default=0)
    parser.add_argument('-n', '--num_samples', type=int, default=2)

    args = parser.parse_args()
    inference(args)
