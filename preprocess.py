#!/usr/bin/env python3
###########################################################
# Authors: Joel Anyanti, Jui-Chieh Chang, Alex Condotti
# Carnegie Mellon Univerity
# 11-785 (Introduction to Deep Learning)
#
# preprocess.py
###########################################################
# Imports
###########################################################
import os
import pretty_midi
import pypianoroll
import numpy as np
import time
###########################################################
# Constants
###########################################################
### Resolution is set to 24 => there are 24 pulses in a quarter note
RESOLUTION = 24
### Time signature is set to 4/4 => there are 24*4 = 96 pulses in a bar
PPB = RESOLUTION * 4
### Time signature is set to 4/4 => there are 4 quarter notes in a bar
QPB = 4
### The pitch of each pulse is represented as a 128-dimensional binary vector
### The chord of each quarter note is represented as a 13-dimensional binary vector

resolutions = list()
CHORD_MAP = {'A':0,'Bb':1,'B':2,'C':3,'C#':4,'D':5,'Eb':6,'E':7,'F':8,'F#':9,'G':10,'G#':11,'Ab':11}

"""
    Transfrom the MIDI data and chords in POP909 into encoded vectors with the following format:
        midi_encodings (numpy.ndarray) (shape=(total_num_of_bars, 128, PPB), dtype=uint8)
        chord_encodings (numpy.narray) (shape=(total_num_of_bars, QPB, 13), dtype=uint8)

"""

def transform(root_dir, output_path):
    """
       Traverse and encode all the MIDI files from root_dir
    """
    # list of numpy.ndarrays(128, 96), each representing the melody of a bar
    midis = []
    # list of numpy.ndarrays(13, 4), each representing the chords of a bar
    chords = []
    # list of nump.ndarrays(13, 1), each representing the chord of a bar
    chords_per_bar = []
    # list of integers, each representing the index of the starting bar of a midi file
    margin_indices = [0]

    paths = sorted([d for d in os.listdir(root_dir) if d.isdecimal()])
    counter = 0
    error_count = 0
    start_time = time.time()
    for path in paths:
        chord_filepath = os.path.join(root_dir, path, "chord_midi.txt")
        midi_filepath = os.path.join(root_dir, path, path+".mid")

        chord_encodings, base_tick = parse_chords(chord_filepath, midi_filepath)
        midi_encodings = parse_midi(midi_filepath, base_tick)

        # check if chord and midi hava the same number of bars
        lc = len(chord_encodings)
        lm = len(midi_encodings)

        # possible misalignment or too few bars
        if abs(lc - lm) > 0 or lm < 20:
            #print("counter = {}, chord bar = {}, midi bar = {}".format(counter, len(chord_encodings), len(midi_encodings)))
            #print("base = {}".format(base_tick))
            error_count += 1

        # only store the aligned midi files
        else:
            bar_count = 0
            for c, m in zip(chord_encodings, midi_encodings):
                # drop empty midi bars
                if m.sum() != 0:
                    midis.append(m) # (13, 4)
                    chords.append(c) # (13, 96)
                    chords_per_bar.append(np.expand_dims(c[:, 0], axis=1)) # (13, 1)
                    bar_count += 1

            margin_indices.append(margin_indices[-1] + bar_count)

        counter += 1

    for c, cpb in zip(chords, chords_per_bar):
        if cpb.shape != (13, 1) or not (cpb[:, 0] == c[:, 0]).all():
            print("somethings wrong")

    np.savez(output_path, midis=np.array(midis).astype(np.uint8),
             chords=np.array(chords).astype(np.uint8),
             chords_per_bar=np.array(chords_per_bar).astype(np.uint8),
             margin_indices=margin_indices)

    print("Took {}s to parse {} midi files, with a total of {} bars".format(time.time()-start_time, counter-error_count, len(chords)))
    print("output is stored at {}.npz".format(output_path))

    # count types of resolutions
    r = np.array(resolutions)
    unique, counts = np.unique(r, return_counts=True)
    #print("Resolutions count:", dict(zip(unique, counts)))

def parse_chords(chord_filepath, midi_filepath):
    """
       Parse and encode the chords in chord_filepath into numpy format
    """
    # load midi files
    midi_data = pretty_midi.PrettyMIDI(midi_filepath, resolution=24)

    # resolution == PPQ == ticks per quarter note
    resolution = midi_data.resolution
    resolutions.append(resolution)
    # time signatures == numerator/denominator
    numerator = midi_data.time_signature_changes[0].numerator
    denominator = midi_data.time_signature_changes[0].denominator
    ticks_per_bar = int(resolution * numerator * 4 / denominator)

    chord_encodings = []

    base_tick = 0
    # read from "chord_midi.txt"
    with open(chord_filepath, 'rt') as f:
        lines = f.readlines()
        # treat the first chord onset as base
        base = midi_data.time_to_tick(float(lines[0].strip().split()[0]))
        # calculate offset for midi file
        base_tick = int(round(base / (resolution/RESOLUTION)))
        for l in lines:
            onset, offset, chord = l.strip().split()
            onset = float(onset)
            offset = float(offset)

            # shift the onset and offset for alignment
            tick_onset = midi_data.time_to_tick(onset) - base
            tick_offset = midi_data.time_to_tick(offset) - base

            # get the number of quarters notes the chord spans
            length = int(round((tick_offset - tick_onset) / resolution))

            #if (tick_offset - tick_onset) % resolution != 0:
            #    print("remainder = {}, resolution = {}".format(tick_offset - tick_onset, resolution))

            # get one-hot representation of the chord
            chord_encodings.append(get_chord_encoding(chord, length))

    # concatenate encodings along the "time" axis so that shape = (13, num_of_quarter_notes)
    chord_encodings = np.concatenate(chord_encodings, axis=1)

    # num_of_bars = num_of_quarter_notes / quarter_notes_per_bar
    num_bars = chord_encodings.shape[1] // QPB

    # group quarter notes into list of bars (List(num_bars): numpy.ndarray(13 x QPB))
    chord_encodings = np.split(chord_encodings[:, :num_bars*QPB], num_bars, axis=1)

    # all numpy arrays should have shape = (13, 4 (QPB))
    for c in chord_encodings:
        if c.shape != (13, 4):
            print("error")

    return chord_encodings, base_tick

def get_chord_encoding(chord, length):
    """
       Get the encoding of the given chord in numpy one-hot format
    """
    encoding = np.zeros((13, length))
    # check if it is major
    encoding[12, :] = 1 if "maj" in chord else 0
    # encode the key
    if 'N' not in chord:
        encoding[CHORD_MAP[chord.split(':')[0]], :] = 1

    return encoding

def parse_midi(midi_filepath, base_tick):
    """
       Parse and encode the MIDI data in midi_filepath into numpy format
    """
    # load MIDI file
    midi_data = pypianoroll.read(path=midi_filepath, resolution=RESOLUTION)

    # stack the tracks and select only the MELODY track
    midi_melody = midi_data.stack()[0] # (Time_step x Pitch_range)

    # set all velocity values to zero to binarize data
    midi_melody[midi_melody >= 1] = 1

    # shift the pulses using base_tick as offset
    midi_melody = midi_melody[base_tick:]

    # num_of_bars = num_of_pulses / pulses_per_bar
    num_bars = midi_melody.shape[0] // PPB

    # transpose so that shape = (Pitch_range x Time_step)
    midi_melody = np.transpose(midi_melody, axes=(1,0))

    # group pulses into list of bars (List(num_bars): numpy.ndarray(Pitch_range x PPB))
    midi_melody = np.split(midi_melody[:, :num_bars*PPB], num_bars, axis=1)

    # all numpy arrays should have shape = (128 (Pitch_range), 96 (PPB))
    for m in midi_melody:
        if m.shape != (128, 96):
            print("error")

    return midi_melody

if __name__ == "__main__":
    root_dir = "./POP909"
    output_path = "./encodings"
    transform(root_dir, output_path)
