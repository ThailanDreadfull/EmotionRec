import numpy as np
from sys import platform
import subprocess

IS_LINUX = True
if platform == "win32":
    IS_LINUX = False

PROJECT_PATH = "/root/dev/emotion_speech_recognition/" if IS_LINUX else "C:\dev\emotion_speech_recognition\\"
PATH_TO_READ_FILES = PROJECT_PATH + ('audio_sample/' if IS_LINUX else 'audio_sample\\')

CMDS = ['SMILExtract', '-C', '', '', '']


# SMILExtract -C features_config_files/demo1.conf -I audio_sample/irritado-female-2.wav -O extracted_features/irritado-female-2.energy.csv


def generate_cmd(filename, feature_name):
    cmd = ['SMILExtract', '-C']

    output_filename = filename.split('.')[0] + '.'+feature_name+'.csv'

    cmd.append( PROJECT_PATH+'features_config_files/demo1.conf' )
    cmd.append( '-I' )
    cmd.append( PATH_TO_READ_FILES+filename )
    cmd.append( '-O' )
    cmd.append( PROJECT_PATH+'extracted_features/'+output_filename )

    return cmd

def execute_cmd(cmd):
    bashCommand = "cwm --rdf test.rdf --ntriples > test.nt"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    return process.communicate()


def extract_pitch(filename):
    # print('Extracting pitch for ', filename)
    cmd = ' '.join( generate_cmd(filename, 'pitch') )
    # print cmd
    # output, error = execute_cmd(cmd)

    return np.array([0,0,0])

def extract_energy(filename):
    print('Extracting energy for ', filename)
    cmd = ' '.join( generate_cmd(filename, 'pitch') )
    return np.array([0,0,0])

def extract_mfcc(filename):
    print('Extracting MFCC for ', filename)
    cmd = ' '.join( generate_cmd(filename, 'pitch') )
    return np.array([0,0,0])

def extract_formants(filename):
    pass

def extract_lpcc(filename):
    pass

def extract_lfpc(filename):
    pass
