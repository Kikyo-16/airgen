import numpy as np

from ..utilities.symbolic_utils import process_midi, process_chord, process_beat
from ..utilities.sep_utils import separate
from ..utilities.encodec_utils import extract_rvq, extract_musicgen_emb
from ..utilities import *
import librosa
from config import *

device = get_device()


def process_single(output_folder, path_dict, fname):
    output_folder = os.path.join(output_folder, fname)
    print("begin", output_folder)
    mkdir(output_folder)
    audio_path = path_dict["audio"]
    midi_path = path_dict["midi"] + ".piano.wav"
    chord_audio_path = path_dict["chord"] + ".mid.wav"
    sr = 48000

    piano_output_path = os.path.join(output_folder, "piano_rvq.npy")
    piano_drums_output_path = os.path.join(output_folder, "piano_drums_rvq.npy")

    #process_midi(midi_path, flatten_midi_output_path=midi_path+".piano.mid")

    drums_output_path = os.path.join(output_folder, "drums_rvq.npy")
    bass_output_path = os.path.join(output_folder, "bass_rvq.npy")
    drums_bass_output_path = os.path.join(output_folder, "drums_bass_rvq.npy")
    other_output_path = os.path.join(output_folder, "other_rvq.npy")
    mix_output_path = os.path.join(output_folder, "mix_rvq.npy")
    chord_audio_output_path = os.path.join(output_folder, "chord_rvq.npy")
    #midi_output_path = os.path.join(output_folder, "midi_rvq.npy")
    print(piano_drums_output_path)
    if os.path.exists(chord_audio_output_path):
        print(chord_audio_output_path, "skip")
        return

    wav, _ = librosa.load(audio_path, sr=sr, mono=True)
    piano, _ = librosa.load(midi_path, sr=sr, mono=True)
    chord_audio, _ = librosa.load(chord_audio_path, sr=sr, mono=True)


    piano = np2torch(piano).to(device)[None, None, ...]
    chord_audio = np2torch(chord_audio).to(device)[None, None, ...]

    wav = np2torch(wav).to(device)[None, None, ...]
    wavs = separate(wav, sr)
    print("separate", output_folder)

    mean = torch.mean(wavs["other"])
    std = torch.std(wavs["other"])

    piano = (piano - torch.mean(piano)) / torch.std(piano)
    piano = piano * std + mean

    chord_audio = (chord_audio - torch.mean(chord_audio)) / torch.std(chord_audio)
    chord_audio = chord_audio * std + mean

    chord_rvq = extract_rvq(chord_audio, sr=sr)

    piano_rvq = extract_rvq(piano, sr=sr)
    '''piano_drums_rvq = extract_rvq(align_sum_fst(piano, wavs["drums"]), sr=sr)



    drums_rvq = extract_rvq(wavs["drums"], sr=sr)
    bass_rvq = extract_rvq(wavs["bass"], sr=sr)
    drums_bass_rvq = extract_rvq((wavs["bass"] + wavs["drums"]) / np.sqrt(2), sr=sr)
    other_rvq = extract_rvq(wavs["other"], sr=sr)
    mix_rvq = extract_rvq(wav, sr=sr)

    np.save(drums_output_path, drums_rvq.cpu().numpy())
    np.save(bass_output_path, bass_rvq.cpu().numpy())
    np.save(drums_bass_output_path, drums_bass_rvq.cpu().numpy())
    np.save(other_output_path, other_rvq.cpu().numpy())
    np.save(mix_output_path, mix_rvq.cpu().numpy())'''

    np.save(piano_output_path, piano_rvq.cpu().numpy())
    #np.save(piano_drums_output_path, piano_drums_rvq.cpu().numpy())
    np.save(chord_audio_output_path, chord_rvq.cpu().numpy())


def scan_audio(audio_folder, low, up):
    res = {}
    for song in os.listdir(audio_folder):
        fname = song.split("Track")[-1]
        audio_path = os.path.join(audio_folder, song, "mix.flac")
        midi_path = os.path.join(audio_folder, song, "all_src.mid")
        if not os.path.exists(audio_path):
            audio_path = os.path.join(audio_folder, song, "mix.mp3")

        if int(fname) < low or int(fname) >= up:
            continue
        chord_path = audio_path + ".chord.lab"
        beat_path = audio_path + ".beat.lab"
        if not os.path.exists(chord_path):
            chord_path = str.replace(chord_path, ".mp3", ".flac")
            beat_path = str.replace(beat_path, ".mp3", ".flac")

        res[fname] = {
            "audio": audio_path,
            "midi": midi_path,
            "chord": chord_path,
            "beat": beat_path
        }
    return res
