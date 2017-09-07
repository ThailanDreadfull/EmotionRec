#!/usr/bin/env python

import os
from sys import platform

import csv_handler
import feature_extractor


IS_LINUX = True
if platform == "win32":
    IS_LINUX = False

PROJECT_PATH = "/root/dev/emotion_speech_recognition/" if IS_LINUX else "C:\dev\emotion_speech_recognition\\"
PATH_TO_READ_FILES = PROJECT_PATH + ('audio_sample/' if IS_LINUX else 'audio_sample\\')


def extract_audio_features_from_path(path):
    filenames = os.listdir(path)

    for filename in filenames:
        print('Extracting features from audio: ', filename)

        feature_extractor.extract_pitch(filename)
        feature_extractor.extract_energy(filename)
        feature_extractor.extract_mfcc(filename)


        pass


if __name__ == '__main__':
    print('Starting program...')

    # PATH_TO_READ_FILES = PROJECT_PATH + 'audio_sample/'

    extract_audio_features_from_path(PATH_TO_READ_FILES)



    csv_handler.import_csv_from_file()

    print('Ending program...')