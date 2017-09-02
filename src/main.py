#!/usr/bin/env python

import sys
#sys.path.append('/root/miniconda2/lib/')
#from foo import util,bar

import os
import argparse

#import numpy as np
#from scipy import special, optimize
#import matplotlib.pyplot as plt

from python_speech_features import mfcc
import scipy.io.wavfile as wav


# All values in ms
WINDOW_SIZE = 25.
WINDOW_STEP = 10.


def extract_mfcc(filename):
    (rate,sig) = wav.read( filename )
    
    print("rt: {}".format( rate ))

    #nfft changed to the same number of frame length, because the frame was being truncated
    mfcc_feat = mfcc(sig, rate, WINDOW_SIZE/1000, WINDOW_STEP/1000, nfft=1103)

    #default inputs (from documentation)
    #mfcc_feat = mfcc(sig,samplerate=16000,winlen=0.025,winstep=0.01,numcep=13,
    #             nfilt=26,nfft=512,lowfreq=0,highfreq=None,preemph=0.97, ceplifter=22,appendEnergy=True)
    
    #not used for now...
    #d_mfcc_feat = delta(mfcc_feat, 2)
    #fbank_feat = logfbank(sig,rate)

    print(mfcc_feat)
#    return mfcc_feat #return breaks I dont know why yet

#import librosa

def extract_energy(filename):
    pass
    
    #mtd from librosa
    #Compute root-mean-square (RMS) energy for each frame, either from the audio samples y or from a spectrogram S.
    #rmse([y, S, frame_length, hop_length, ...])

def extract_pitch(filename):
    print( 'Extracting pitch for {}'.format(filename) )

    #I've tried to use pyacoustics without success
    #pyacoustics.intensity_and_pitch.praat_pi.getPraatPitchAndIntensity()

    # Parse command-line arguments
    parser = argparse.ArgumentParser(usage=__doc__)
    parser.add_argument("--order", type=int, default=3, help="order of Bessel function")
    parser.add_argument("--output", default="plot.png", help="output image file")
    args = parser.parse_args()

    # Compute maximum
    f = lambda x: -special.jv(args.order, x)
    sol = optimize.minimize(f, 1.0)

    # Plot
    x = np.linspace(0, 10, 5000)
    plt.plot(x, special.jv(args.order, x), '-', sol.x, -sol.fun, 'o')

    # Produce output
    plt.savefig(args.output, dpi=96)

    





    return []

def read_all_audio_files_from_path( path ):
    ret = os.listdir(path)

    filenames = []
    for name in ret:
        filenames.append(path+name)

    return filenames

if __name__ == '__main__':
    print('Starting program...')

    path_to_read_files = '/root/dev/emotion_speech_recognition/audio_sample/'
    # path_to_read_files = 'C:\dev\emotion_speech_recognition\\audio_sample\\'
    filenames = read_all_audio_files_from_path(path_to_read_files)

    for filename in filenames:
 #       pitch = extract_pitch(filename)
        extract_mfcc(filename)

    # extract_parameters()
    # extract_chromagram("/root/tcc/extraction/audio_samples/teste2-2.wav")
    
    print('Ending program...')


