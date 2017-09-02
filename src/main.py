#!/usr/bin/env python

#import numpy as np
#from hmmlearn import hmm
#np.random.seed(42)
#
#model = hmm.GaussianHMM(n_components=3, covariance_type="full")
#model.startprob_ = np.array([0.6, 0.3, 0.1])
#model.transmat_ = np.array([[0.7, 0.2, 0.1],
#                            [0.3, 0.5, 0.2],
#                            [0.3, 0.3, 0.4]])

#model.means_ = np.array([[0.0, 0.0], [3.0, -3.0], [5.0, 10.0]])
#model.covars_ = np.tile(np.identity(2), (3, 1, 1))
#X, Z = model.sample(100)


#lr = hmm.GaussianHMM(n_components=3, covariance_type="diag", init_params="cm", params="cmt")
#lr.startprob_ = np.array([1.0, 0.0, 0.0])
#lr.transmat_ = np.array([[0.5, 0.5, 0.0],
#                          [0.0, 0.5, 0.5],
#                          [0.0, 0.0, 1.0]])
#





#
# import subprocess
# from bregman.suite import Chromagram


# def extract_pitch( audio_file_path ):
#     # Read in a WAV and find the freq's
#     import pyaudio
#     import wave
#     import numpy as np
#
#     chunk = 2048
#
#     # open up a wave
#     wf = wave.open(audio_file_path, 'rb')
#     swidth = wf.getsampwidth()
#     RATE = wf.getframerate()
#     # use a Blackman window
#     window = np.blackman(chunk)
#     # open stream
#     p = pyaudio.PyAudio()
#     stream = p.open(format=
#                     p.get_format_from_width(wf.getsampwidth()),
#                     channels=wf.getnchannels(),
#                     rate=RATE,
#                     output=True)
#
#     # read some data
#     data = wf.readframes(chunk)
#     # play stream and find the frequency of each chunk
#     while len(data) == chunk * swidth:
#         # write data out to the audio stream
#         stream.write(data)
#         # unpack the data and times by the hamming window
#         indata = np.array(wave.struct.unpack("%dh" % (len(data) / swidth), data)) * window
#         # Take the fft and square each value
#         fftData = abs(np.fft.rfft(indata)) ** 2
#         # find the maximum
#         which = fftData[1:].argmax() + 1
#         # use quadratic interpolation around the max
#         if which != len(fftData) - 1:
#             y0, y1, y2 = np.log(fftData[which - 1:which + 2:])
#             x1 = (y2 - y0) * .5 / (2 * y1 - y2 - y0)
#             # find the frequency and output it
#             thefreq = (which + x1) * RATE / chunk
#             print("The freq is %f Hz." % (thefreq))
#         else:
#             thefreq = which * RATE / chunk
#             print("The freq is %f Hz." % (thefreq))
#         # read some more data
#         data = wf.readframes(chunk)
#     if data:
#         stream.write(data)
#     stream.close()
#     p.terminate()







# def extract_chromagram(audio_file_path):
#
#
#     F = Chromagram(audio_file_path, nfft=16384, wfft=8192, nhop=2205)
#     all_chroma_ft = F.X # all chroma features
#     one_ft = F.X[:,0] # one feature
#
#     print('All chrome features: '+ str(all_chroma_ft))
#     print('One feature: '+ str(one_ft))





# def extract_parameters():
#     cmd = []
#     cmd.append('yaafe')
#     cmd.append('-v')
#     cmd.append('-r')
#     cmd.append('44100')
#     cmd.append('-o')
#     cmd.append('csv')
#     cmd.append('-i')
#     cmd.append('/root/tcc/extraction/to_extract')
#     cmd.append('-b')
#     cmd.append('/root/tcc/extraction/output')
#     #cmd.append('-f')
#     #cmd.append('Energy:Derivate')
#     cmd.append('-c')
#     cmd.append('/root/tcc/extraction/featurePlan')
#
#     subprocess.check_output(cmd)
#
#     print("Command: "+" ".join(cmd))
#
#     output, err  = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).communicate()
#
#     if not err:
#         print('>>>>>>>>>>>>>>>>>>>>>>>>')
#         print(output)
#         print('>>>>>>>>>>>>>>>>>>>>>>>>')
#     else:
#         print('>>>>>>>>>>> error <<<<<<<<<<<<')
#         print(err)
#         print('>>>>>>>>>>>>>><<<<<<<<<<<<<<<<')

import os
# import pyacoustics
import argparse

import numpy as np
from scipy import special, optimize
import matplotlib.pyplot as plt

from python_speech_features import mfcc
#from python_speech_features import delta
#from python_speech_features import logfbank
import scipy.io.wavfile as wav

# All values in ms
WINDOW_SIZE = 25.
WINDOW_STEP = 10.


def extract_mfcc(filename):
    (rate,sig) = wav.read( filename )
    
    #print("Signal: {}".format( WINDOW_SIZE/1000 ))

    #nfft changed to the same number of frame length, because the frame was been truncated
    mfcc_feat = mfcc(sig, rate, WINDOW_SIZE/1000, WINDOW_STEP/1000, nfft=1103)

    #default inputs (from documentation)
    #mfcc_feat = mfcc(sig,samplerate=16000,winlen=0.025,winstep=0.01,numcep=13,
    #             nfilt=26,nfft=512,lowfreq=0,highfreq=None,preemph=0.97, ceplifter=22,appendEnergy=True)
    
    #not used for now...
    #d_mfcc_feat = delta(mfcc_feat, 2)
    #fbank_feat = logfbank(sig,rate)

    print(mfcc_feat)
#    return mfcc_feat #return breaks I dont know why yet



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


