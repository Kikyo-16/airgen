import pretty_midi
import numpy as np
from .chord import encode


def load_chord_dict(path):
    with open(path, "r") as f:
        sc = f.readlines()
    sc = [s.rstrip().split("\t") for s in sc]
    sc_dict = {}
    desc_dict = {}
    matrix_dict = {}
    for s in sc:
        sc_dict[s[0]] = int(s[1])
        desc_dict[s[1]] = s[0]

        out = np.zeros([36])
        chs = encode(s[0])
        out[chs[0]] = 1
        out[chs[2] + 12] = 1
        for j in range(12):
            out[j + 24] = chs[1][j]
        matrix_dict[s[1]] = out
    sc_dict["N"] = 0
    desc_dict["0"] = "N"
    return sc_dict, desc_dict, matrix_dict


def convert_simple_chord(chords):
    outs = []
    tags = ["maj", "min", "aug", "dim", "hdim", "hdim7", "sus2", "sus4"]
    for chord in chords:
        ch = chord[-1]
        if ch == "N":
            continue
        key, sc = ch.split(":")
        ch = None
        for tag in tags:
            if str.startswith(sc, tag):
                ch = key + ":" + tag
        if ch is None:
            ch = key + ":maj"
        outs.append([chord[0], chord[1], ch])
    return outs


def chords_to_matrix(chords, res=50):
    max_len = round(max([ch[1] for ch in chords]) * res + 1)
    out = np.zeros([max_len, 12 + 12 + 12], dtype=np.int16)
    for chord in chords:
        ch = chord[-1]
        if ch == "N":
            continue
        st = int(float(chord[0]) * res)
        ed = round(float(chord[1]) * res)
        # print(ch)
        chs = encode(ch)
        out[st:] = 0
        out[st: ed, chs[0]] = 1
        out[st: ed, chs[2] + 12] = 1
        # print(ch, st, ed, chs)
        for j in range(12):
            out[st: ed, j + 24] = chs[1][j]
    return out


def chords_to_midi(chords, path):
    midi = pretty_midi.PrettyMIDI()
    instr = pretty_midi.Instrument(0)

    for chord in chords:
        ch = chord[-1]
        if ch == "N":
            continue
        st = float(chord[0]) / 3
        ed = float(chord[1]) / 3
        ch_ns = encode(ch)

        root = ch_ns[0]
        lowest = ch_ns[2]
        ch = ch_ns[1]
        outs = []
        for i in range(0, 12):
            if ch[i] == 1:
                note = pretty_midi.Note(velocity=100,
                                        pitch=i + root + lowest + 12 * 5,
                                        start=st,
                                        end=ed)
                instr.notes.append(note)
                outs.append(i + root + lowest + 12 * 5)

        print(outs)

    midi.instruments.append(instr)
    midi.write(path)


def chord_to_notes(ch):
    ch_ns = encode(ch)
    root = ch_ns[0]
    lowest = ch_ns[2]
    ch = ch_ns[1]
    outs = []
    for i in range(0, 12):
        if ch[i] == 1:
            p = i + root + 12 if i < lowest else i + root
            outs.append(p + 12 * 5)
    return outs
