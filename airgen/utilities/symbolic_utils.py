import sys
import os
import numpy as np
import pretty_midi
import torch

sys.path.insert(0, os.path.join(sys.path[0], "../../.."))

from .utils import read_lst
from .chord_utils import chords_to_matrix, chord_to_notes
from .reverse_pianoroll import piano_roll_to_pretty_midi


def add_ch(instr, pre, ed, ch):
    assert pre < ed
    if ch == "N":
        return
    notes = chord_to_notes(ch)
    for note_number in notes:
        instr.notes.append(pretty_midi.Note(
            velocity=100, pitch=note_number, start=pre, end=ed))


def chord_with_beat(chord_path, beat_path, midi_path):
    lines = read_lst(chord_path, "\t")
    lines = [[float(line[0]), float(line[1]), line[-1]] for line in lines]
    midi = pretty_midi.PrettyMIDI()
    instr = pretty_midi.Instrument(program=0)
    beats = read_lst(beat_path)
    beats = np.array([float(b.split("\t")[0]) for b in beats])
    assert len(beats) > 0
    st, ed, ch = lines[0]
    cur = 0
    pre = st
    for i in range(len(beats)):
        if beats[i] >= ed:
            while beats[i] >= ed:
                add_ch(instr, pre, ed, ch)
                cur += 1
                if cur >= len(lines):
                    break
                st, ed, ch = lines[cur]
                pre = st
            if cur >= len(lines):
                break
        else:
            add_ch(instr, pre, ed, ch)
            pre = beats[i]
    if cur < len(lines) and ed > pre:
        add_ch(instr, pre, ed, ch)
    midi.instruments.append(instr)
    midi.write(midi_path)


def process_chord_onsets(chords, lines, res):
    onsets = np.zeros_like(chords[:, :3])

    for line in lines:
        st = round(line[0] * res)
        ed = round(line[1] * res)
        if line[-1] == "N":
            onsets[st: ed, 2] = 1
        else:
            onsets[st: ed, 1] = 1
        onsets[st, :] = 0
        onsets[st, 0] = 1
    return onsets


def process_chord(chord_path, res=50, sec=20, chord_output_path=None, offset=None):
    lines = read_lst(chord_path, "\t")
    lines = [[float(line[0]), float(line[1]), line[-1]] for line in lines]

    if offset is None:
        chords = chords_to_matrix(lines, res=res)
        onsets = process_chord_onsets(chords, lines, res)
        return chords, onsets

    st = None
    for j, line in enumerate(lines):
        if st is None and float(line[1]) > offset:
            st = j
        if st is not None and float(line[1]) >= offset + sec:
            ed = j + 1
            break
    lines = lines[st: ed]
    chords = [[float(ch[0]) - offset, float(ch[1]) - offset, ch[2]] for ch in lines]
    chords[0][0] = 0.01 if chords[0][0] < 0 else chords[0][0]
    chords[-1][1] = sec if chords[-1][1] > sec else chords[-1][1]
    chords = [str(ch[0]) + "\t" + str(ch[1]) + "\t" + ch[2] for ch in chords]

    with open(chord_output_path, "w") as f:
        f.write("\n".join(chords))

    offset = int(offset * res)
    chords = chords_to_matrix(lines, res=res)
    onsets = process_chord_onsets(chords, lines, res)
    return chords[offset: offset + sec * res], onsets[offset: offset + sec * res]


def process_midi(midi_path, midi_output_path=None, flatten_midi_output_path=None, res=50, sec=None, offset=None):
    midi = pretty_midi.PrettyMIDI(midi_path)
    n_midi = pretty_midi.PrettyMIDI()
    for instr in midi.instruments:
        if not instr.is_drum:
            flag = False
            # for tag in ["DRUMS", "SQ-", "SYNTH", 'KIRA']:
            #    if len(str.split(instr.name, tag)) > 1:
            #        flag = True
            if flag:
                continue
            n_midi.instruments.append(instr)
    midi = n_midi

    if offset is not None:
        if sec is not None:
            midi.adjust_times([offset, offset + sec], [0, sec])
        else:
            ed_sec = midi.get_end_time()
            midi.adjust_times([offset, ed_sec], [0, ed_sec - offset])
    if midi_output_path is not None:
        midi.write(midi_output_path)

    piano_roll = midi.get_piano_roll(fs=res)
    piano_roll[piano_roll > 0] = 1
    piano_roll = np.transpose(piano_roll, (1, 0))
    pr = piano_roll

    onsets = np.zeros_like(pr)
    for trk in midi.instruments:
        for event in trk.notes:
            st = round(event.start * res)
            if st >= len(onsets):
                st = st - 1
                print(st, len(onsets))
            onsets[st, event.pitch] = 1

    piano_roll = np.transpose(piano_roll, (1, 0)).astype(np.int16) * 100
    midi = piano_roll_to_pretty_midi(piano_roll, fs=50, program=0)
    if flatten_midi_output_path is not None:
        midi.write(flatten_midi_output_path)
    return pr, onsets


def process_beat(beat_path, max_len=None, res=50, unit=1., split=" "):
    lines = read_lst(beat_path, split=split)
    lines = [round(float(line[0].split("\t")[0]) * res / unit) for line in lines]
    beats = np.zeros([lines[-1] + 1]) if max_len is None else np.zeros([max_len])
    for line in lines:
        beats[line] = 1
    return beats


def reduce_piano(midi_path, reduced_path, res=100):
    midi = pretty_midi.PrettyMIDI(midi_path)
    n_midi = pretty_midi.PrettyMIDI()
    n_instr = pretty_midi.Instrument(0)
    notes = []
    xlen = 0
    for instr in midi.instruments:
        if instr.is_drum:
            continue
        prog = instr.program
        if prog <33 or (105 <= prog <= 109) or prog == 115:
            for n in instr.notes:
                notes.append(n)
                if n.end > xlen:
                    xlen = n.end
    sorted(notes, key=lambda x: -(x.end - x.start))
    cover = np.zeros([round(xlen * res), 128])
    for n in notes:
        st = int(n.start * res)
        ed = int(n.end * res)
        p = n.pitch
        if cover[st:ed, p].sum() == 0:
            cover[st:ed, p] = 1
            n_instr.notes.append(n)
    n_midi.instruments.append(n_instr)
    n_midi.write(reduced_path)


