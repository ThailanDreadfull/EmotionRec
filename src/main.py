#!/usr/bin/env python


from __future__ import print_function

import sys
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

#### FUNCIONANDO OK
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




import yaafelib
import h5py

def extract_energy_with_yaafe(filename):
    
    #yaafe -r 44100 -f "mfcc: MFCC blockSize=1024 stepSize=512" test.wav
    h5py.run_tests()
    #f = h5py.File('myfile.hdf5','r')


import librosa
def extract_energy(filename): #WORKING FUCKKKKKKKKK!!!!!!!
    
    # 1ms = 22 samples, so 25ms = 550 samples
    sig, sample_rate = librosa.load( filename )
    
    #Compute root-mean-square (RMS) energy for each frame, either from the audio samples y or from a spectrogram S.
    energy = librosa.feature.rmse( y=sig, frame_length=550 )
    #librosa.rmse([audio_signal, spectrogram_magnitude, frame_length, hop_length, ...])
    
    return energy


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


import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from hmmlearn import hmm





def training_model_s( features ):
    import numpy as np
    from hmmlearn import hmm

    model = hmm.MultinomialHMM(n_components=3)
    model.startprob_ = np.array([0.3, 0.4, 0.3])
    model.transmat_ = np.array([[0.2, 0.6, 0.2],
                                [0.4, 0.0, 0.6],
                                [0.1, 0.2, 0.7]])
    model.emissionprob_ = np.array([[0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                                    [0.1, 0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1],
                                    [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2]])

    # Predict the optimal sequence of internal hidden state
    X = np.atleast_2d([3, 4, 5, 6, 7]).T
    print(model.decode(X))



def training_model( features ):
    # import warnings
    # warnings.filterwarnings('ignore')


    energy = np.array(features[0])
    import copy
    energy_2 = copy.deepcopy(energy)


    # quotes = quotes_historical_yahoo_ochl(
    #     "INTC", datetime.date(1995, 1, 1), datetime.date(2012, 1, 6))

    # Unpack quotes
    # dates = np.array([q[0] for q in quotes], dtype=int)
    # close_v = np.array([q[2] for q in quotes])
    # volume = np.array([q[5] for q in quotes])[1:]

    # Take diff of close value. Note that this makes
    # ``len(diff) = len(close_t) - 1``, therefore, other quantities also
    # need to be shifted by 1.
    # diff = np.diff(close_v)
    # dates = dates[1:]
    # close_v = close_v[1:]


    # Pack energy and other features for training.
    X = np.column_stack([energy, energy_2])
    print(X)

    print("fitting to HMM and decoding ...", end="")

    # Make an HMM instance and execute fit
    model = hmm.GaussianHMM(n_components=4, covariance_type="diag", n_iter=1000).fit(X, 1)

    # Predict the optimal sequence of internal hidden state
    hidden_states = model.predict(X)
    print(hidden_states)


    print("done")








    # model = hmm.GaussianHMM(n_components=1, covariance_type="full")
    # model.energy_ = energy
    #
    # X, Z = model.sample(500)
    # print(X)
    # print(Z)







    ################### Printing result #################
    print("Transition matrix")
    print(model.transmat_)
    print()

    print("Means and vars of each hidden state")
    for i in range(model.n_components):
        print("{0}th hidden state".format(i))
        print("mean = ", model.means_[i])
        print("var = ", np.diag(model.covars_[i]))
        print()
    ######################################################

    ################# Plot the sampled data #############
    # plt.plot(X[:, 0], X[:, 1], ".-", label="observations", ms=6, mfc="orange", alpha=0.7)
    #
    # # Indicate the component numbers
    # for i, m in enumerate(means):
    #     print("{}  {}".format(i, m))
    #     plt.text(m[0], m[1], 'Component %i' % (i + 1), size=17, horizontalalignment='center',
    #              bbox=dict(alpha=.7, facecolor='w'))
    #
    # plt.legend(loc='best')
    # plt.show()
    #
    # plt.savefig('foo.png')
    ######################################################


def training_model_old( features ):

    ############### old ####################
    np.random.seed(42)
    #model = hmm.GaussianHMM(n_components=3, covariance_type="full")
    #model.startprob_ = np.array([0.6, 0.3, 0.1])
    #model.transmat_ = np.array([[0.7, 0.2, 0.1], [0.3, 0.5, 0.2], [0.3, 0.3, 0.4]])
    #model.means_ = np.array([[0.0, 0.0], [3.0, -3.0], [5.0, 10.0]])
    #model.covars_ = np.tile(np.identity(2), (3, 1, 1))
    #X, Z = model.sample(500)

    #print(X)
    #print(Z)

    ########################################
    

    startprob = np.array([0.6, 0.3, 0.1, 0.0])
    # The transition matrix, note that there are no transitions possible
    # between component 1 and 3
    transmat = np.array([[0.7, 0.2, 0.0, 0.1],
                         [0.3, 0.5, 0.2, 0.0],
                         [0.0, 0.3, 0.5, 0.2],
                         [0.2, 0.0, 0.2, 0.6]])

    # The means of each component
    means = np.array([[0.0,  0.0],
                      [0.0, 11.0],
                      [9.0, 10.0],
                      [11.0, -1.0]])

    # The covariance of each component
    covars = .5 * np.tile(np.identity(2), (4, 1, 1))

    # Build an HMM instance and set parameters
    model = hmm.GaussianHMM(n_components=4, covariance_type="full")

    # Instead of fitting it from the data, we directly set the estimated
    # parameters, the means and covariance of the components
    model.startprob_ = startprob
    model.transmat_ = transmat
    model.means_ = means
    model.covars_ = covars


    X, Z = model.sample(500)
    #print(Z)
    print(X)
    
    # Plot the sampled data
    plt.plot(X[:, 0], X[:, 1], ".-", label="observations", ms=6, mfc="orange", alpha=0.7)

    # Indicate the component numbers
    for i, m in enumerate(means):
        print("{}  {}".format(i, m))
        plt.text(m[0], m[1], 'Component %i' % (i + 1), size=17, horizontalalignment='center', 
                 bbox=dict(alpha=.7, facecolor='w'))

    plt.legend(loc='best')
    plt.show()

    plt.savefig('foo.png')


def test():

    #from __future__ import print_function

    import datetime

    import numpy as np
    from matplotlib import cm, pyplot as plt
    from matplotlib.dates import YearLocator, MonthLocator
    try:
        from matplotlib.finance import quotes_historical_yahoo_ochl
    except ImportError:
        # For Matplotlib prior to 1.5.
        from matplotlib.finance import ( quotes_historical_yahoo as quotes_historical_yahoo_ochl )

    from hmmlearn.hmm import GaussianHMM


    print(__doc__)






    print("fitting to HMM and decoding ...", end="")

    # Make an HMM instance and execute fit
    model = GaussianHMM(n_components=4, covariance_type="diag", n_iter=1000).fit(X)

    # Predict the optimal sequence of internal hidden state
    hidden_states = model.predict(X)

    print("done")







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
        ftrs = []
        ftrs.append( extract_energy(filename) )

        #pitch = extract_pitch(filename)
        #extract_mfcc(filename)

        #print( ftrs )

        training_model( ftrs )
        # test()

        break

    # extract_parameters()
    # extract_chromagram("/root/tcc/extraction/audio_samples/teste2-2.wav")
    
    print('Ending program...')

