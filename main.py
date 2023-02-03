import os
import pretty_midi
import pypianoroll
import numpy as np
import time
import torch
import random, string
from model import Generator, Discriminator
from preprocess import parse_midi
import streamlit as st

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
st.title("MIDInet")
st.write("")
###########################################################
# Generate Sequences
###########################################################
def generate(model, data, nz, num_bars=16, amplify=True):
    model.eval()
    output_sequence = []

    for bar in range(num_bars):
        noise = torch.randn(1, nz).cpu()
        if bar == 0:
            prev = data
        else:
            prev = output_sequence[-1]

        next_bar = model(noise, prev)
        next_bar = next_bar.permute(0, 1, 3, 2)
        # enhance the maximum pitch and drop the rest for each pulse (monophonic)
        if amplify:
            index_arr = torch.max(next_bar[0, 0], axis=0)[1]
            amplified_bar = torch.zeros(next_bar.size()).cpu()
            for i, idx in enumerate(index_arr):
                # only keep the pitch that has velocity > 0.6
                if next_bar[0, 0, idx, i] > 0.6:
                    amplified_bar[0, 0, idx, i] = 1
            #amplified_bar[0, 0, index_arr, np.arange(index_arr.size(0))] = 1
            output_sequence.append(amplified_bar)
        else:
            output_sequence.append(next_bar)

    return output_sequence

def amplify_bar(bar, threshold = 0.65, cuda = True):
    """
       Binarize and amplify the encoding of the given bar
          bar (torch.Float(1 x 1 x 128 x 96))
          threshold (float)
    """
    # get the maximum entry of each timestep
    index_arr = torch.max(bar[0, 0], axis=0)[1]

    amplified_bar = torch.zeros(bar.size()).cuda()

    for i, idx in enumerate(index_arr):
        # only keep the pitch that has velocity > threshold
        if bar[0, 0, idx, i] > threshold:
            amplified_bar[0, 0, idx, i] = 1

    return amplified_bar

def write_midi(sequence, output_path):
    """
       Transform the given sequence into MIDI format and store it in the given path
          sequence (List(num_bars): torch.Tensor(1 x 1 x 128 x 96)): sequence of bar encodings
          output_path (str): path to store the transformed MIDI
    """
    # get number of bars
    num_bars = len(sequence)

    # squeeze into List(num_bars): numpy.ndarray(128 x 96)
    melody = [sequence[i][0][0].detach().cpu().numpy() for i in range(num_bars)]

    # transform into MIDI track format (num_bars*RESOLUTION x 128)
    melody = np.concatenate(melody, axis=1).transpose()

    # pack into binary track
    melody_track = pypianoroll.BinaryTrack(pianoroll = melody > 0)

    # pack into multi-track
    multi_track = pypianoroll.Multitrack(resolution=RESOLUTION, tracks=[melody_track])

    # write to output path
    pypianoroll.write(output_path, multi_track)

def randomword(length):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))


device = torch.device("cpu")
lr_scale_g = 0.25
betas = (0.5, 0.99)
epochs = 20
nz = 30
is_chord = False
chord_dims = (13,1)

model = Generator(nz=nz, is_chord=is_chord, chord_dims=chord_dims)
model.to(device)
st.text("initiated model")

model.load_state_dict(torch.load('g.pth', map_location=torch.device('cpu')))
st.text("loaded model")
model.eval()

uploaded_file = st.file_uploader("Choose a file")
base_tick = st.number_input("Base tick", step=1)

if (st.button('Submit')):
    fn = './' + randomword(20) + '.mid'
    f = open(fn, 'wb')
    f.write(uploaded_file.getvalue())
    f.close()

    data = parse_midi(fn, int(base_tick))
    data = np.reshape(data[0], (1,1,128,96)).astype(np.float32)
    os.remove(fn)

    seq = generate(model, torch.from_numpy(data).cpu(), nz)
    st.text("generated midi")
    write_midi(seq, fn)
    audio_file = open(fn, 'rb')
    audio_bytes = audio_file.read()

    st.text("written midi")
    st.download_button('Download', audio_bytes, file_name='main.mid')
    os.remove(fn)
