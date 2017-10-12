#!/usr/bin/env python

import os
from sys import platform
from pyAudioAnalysis import audioTrainTest as aT
import argparse
import time
import signal
import sys
sys.path.append('C:\dev\emotion_speech_recognition\src\dependencies\pyAudioAnalysis')


#Nao mexer dessa linha pra baixo, pq soh deus sabe como funciona essa bosta

IS_LINUX = True
VERBOSE = True
if platform == "win32":
    IS_LINUX = False


def parallel_method():
    print "entrou no paralelo"

    samples_prefix = "/mnt/c/dev/emotion_speech_recognition/audio_samples/german_emo/"

    SHORT_TERM_STEP = 0.015
    SHORT_TERM_WINDOW = 0.027
    MID_TERM_STEP = 0.85
    MID_TERM_WINDOW = 1.45

    percent_train = 0.70

    list_of_dirs_or_classes = []
    dirs = [d for d in os.listdir(samples_prefix) if os.path.isdir(os.path.join(samples_prefix, d))]
    for dire in dirs:
        list_of_dirs_or_classes.append(samples_prefix + str(dire))


    # aT.featureAndTrain(list_of_dirs_or_classes, MID_TERM_WINDOW, MID_TERM_STEP, SHORT_TERM_WINDOW, SHORT_TERM_STEP, "knn", "models/KNN_MODEL", False, perTrain=percent_train, confusionMatrixInPercent=False, verbosity=False)
    # aT.featureAndTrain(list_of_dirs_or_classes, MID_TERM_WINDOW, MID_TERM_STEP, SHORT_TERM_WINDOW, SHORT_TERM_STEP, "svm", "models/SVM_MODEL", False, perTrain=percent_train, confusionMatrixInPercent=False, verbosity=False)
    # aT.featureAndTrain(list_of_dirs_or_classes, MID_TERM_WINDOW, MID_TERM_STEP, SHORT_TERM_WINDOW, SHORT_TERM_STEP, "extratrees", "models/TREES_MODEL", False, perTrain=percent_train, confusionMatrixInPercent=False, verbosity=False)
    # aT.featureAndTrain(list_of_dirs_or_classes, MID_TERM_WINDOW, MID_TERM_STEP, SHORT_TERM_WINDOW, SHORT_TERM_STEP, "gradientboosting", "models/GRAD_MODEL", False, perTrain=percent_train, confusionMatrixInPercent=False, verbosity=False)
    aT.featureAndTrain(list_of_dirs_or_classes, MID_TERM_WINDOW, MID_TERM_STEP, SHORT_TERM_WINDOW, SHORT_TERM_STEP, "randomforest", "models/FOREST_MODEL", False, perTrain=percent_train, confusionMatrixInPercent=False, verbosity=False)


def print_parameters(st_w, st_s, mt_w, mt_s, perc_train, use_svm, model_name):
    print "\n=============== {} ================".format( "SVM" if use_svm else "KNN")
    print "Parameters for model '{}'".format(model_name)
    print
    print "Short term window: {}".format(st_w)
    print "Short term step: {}".format(st_s)
    print "Mid term window: {}".format(mt_w)
    print "Mid term step: {}".format(mt_s)
    print "Utilizing {:.0f}% of samples for training".format( perc_train * 100 )
    print "===================================="


def feature_and_train_multiclassifier(samples_prefix, st_w, st_s, mt_w, mt_s, perc_train, confusion_matrix_perc, use_svm, verbosity):

    list_of_dirs_or_classes = []
    dirs_models = [d for d in os.listdir(samples_prefix) if os.path.isdir(os.path.join(samples_prefix, d))]
    for dire in dirs_models:

        dirs_classes = [d for d in os.listdir(samples_prefix+dire) if os.path.isdir(os.path.join(samples_prefix+dire, d))]
        for direc in dirs_classes:
            list_of_dirs_or_classes.append(samples_prefix+dire+"/"+direc)

        print "Starting training..."
        start = time.time()
        if use_svm:
            model_name = "models/svm_"+list_of_dirs_or_classes[1].split("/")[-2]+"_"+list_of_dirs_or_classes[1].split("/")[-1]
            print_parameters(st_w, st_s, mt_w, mt_s, perc_train, use_svm, model_name)
            time.sleep(5)
            aT.featureAndTrain(list_of_dirs_or_classes, mt_w, mt_s, st_w, st_s, "svm", model_name, False, perc_train, confusion_matrix_perc)
        else:
            model_name = "models/knn_" + list_of_dirs_or_classes[1].split("/")[-2] + "_" + list_of_dirs_or_classes[1].split("/")[-1]
            print_parameters(st_w, st_s, mt_w, mt_s, perc_train, use_svm, model_name)
            time.sleep(5)
            aT.featureAndTrain(list_of_dirs_or_classes, mt_w, mt_s, st_w, st_s, "knn", model_name, False, perc_train, confusion_matrix_perc)

        end = time.time()
        print "Finished in {:10.2f} seconds".format(end-start)
        time.sleep(10)

        list_of_dirs_or_classes = []
        return 0


def feature_and_train(samples_prefix, st_w, st_s, mt_w, mt_s, perc_train, confusion_matrix_perc, use_svm, verbosity):
    start = time.time()

    list_of_dirs_or_classes = []
    dirs = [d for d in os.listdir(samples_prefix) if os.path.isdir(os.path.join(samples_prefix, d))]
    for dire in dirs:
        list_of_dirs_or_classes.append(samples_prefix + str(dire))

    if use_svm:
        print "Starting training..."

        model_name = "models/SVM_all_classes"

        print_parameters(st_w, st_s, mt_w, mt_s, perc_train, use_svm, model_name)
        bestParam, bestAcc = aT.featureAndTrain(list_of_dirs_or_classes, mt_w, mt_s, st_w, st_s, "svm", model_name, False, perc_train, confusion_matrix_perc, verbosity=verbosity)
        # svmModel = aT.trainSVM(features, CostParam)
    else:
        print "Starting training..."

        model_name = "models/KNN_all_classes"

        print_parameters(st_w, st_s, mt_w, mt_s, perc_train, use_svm, model_name)
        bestParam, bestAcc = aT.featureAndTrain(list_of_dirs_or_classes, mt_w, mt_s, st_w, st_s, "knn", model_name, False, perc_train, confusion_matrix_perc, verbosity=verbosity)
        # knnModel = aT.trainKNN(features, K)

    end = time.time()
    print "Finished in {:10.2f} seconds".format(end-start)
    return bestAcc


def test_file_multiclassifier(filename_to_test, use_svm=True):

    if os.path.isfile(filename_to_test):
        start = time.time()
        if use_svm:
            print "----- SVM -----"
            print "Starting classifier..."
            r, P, classNames = aT.fileClassification(filename_to_test, "models/SVM_MODEL", "svm")
            print "Finished!"
        else:
            print "----- KNN -----"
            print "Starting classifier..."
            r, P, classNames = aT.fileClassification(SAMPLES_PREFIX+"anger/joe.wav_anger_43.80_45.33.wav", "models/SVM_MODEL", "svm")
            print "Finished!"

        chosen = 0.0
        chosenClass = ""
        if len(P) == len(classNames):
            for i in range(0, len(P), 1):

                if P[i] > chosen:
                    chosen = P[i]
                    chosenClass = classNames[i]

        end = time.time()
        print "\n\nThe audio file was classified as {} with prob {}% in {:10.2f} seconds\n\n".format(chosenClass, round(chosen*100, 2), end - start )
    else:
        print "File doesnt exists: {}".format(filename_to_test)


def test_file(filename_to_test, use_svm=True):

    if os.path.isfile(filename_to_test):
        start = time.time()
        if use_svm:
            print "----- SVM -----"
            print "Starting classifier..."
            r, P, classNames = aT.fileClassification(filename_to_test, "models/SVM_MODEL", "svm")
            print "Finished!"
        else:
            print "----- KNN -----"
            print "Starting classifier..."
            r, P, classNames = aT.fileClassification(SAMPLES_PREFIX+"anger/joe.wav_anger_43.80_45.33.wav", "models/SVM_MODEL", "svm")
            print "Finished!"

        chosen = 0.0
        chosenClass = ""
        if len(P) == len(classNames):
            for i in range(0, len(P), 1):

                if P[i] > chosen:
                    chosen = P[i]
                    chosenClass = classNames[i]

        end = time.time()
        print "\n\nThe audio file was classified as {} with prob {}% in {:10.2f} seconds\n\n".format(chosenClass, round(chosen*100, 2), end - start )
    else:
        print "File doesnt exists: {}".format(filename_to_test)


# def signal_handler(signal, frame):
#     print "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa   {}   {}".format(signal, frame)

def train_models(multiclassifier):


    # signal.signal(signal.SIGINT, signal_handler)


    min_st   = 0.020
    max_st   = 0.100
    step_st  = 0.001
    st_overl = 0.4

    min_mt   = 1.200
    max_mt   = 2.1
    step_mt  = 0.100
    mt_overl = 0.5


    MID_TERM_WINDOW = min_mt
    MID_TERM_STEP = round(MID_TERM_WINDOW*mt_overl, 3)
    SHORT_TERM_WINDOW = min_st
    SHORT_TERM_STEP = round(SHORT_TERM_WINDOW*st_overl, 3)

    # # Melhor
    # # precisao: 94.8990825687
    # SHORT_TERM_WINDOW = 0.036
    # SHORT_TERM_STEP = 0.012
    # MID_TERM_WINDOW = 1.3
    # MID_TERM_STEP = 0.65

    confusion_matrix_perc = True
    use_svm = True
    # use_svm = False
    perc_train = 0.75
    VERBOSITY = False

    bestAcc = 0.0
    bestAccParams = {
        "st_w": SHORT_TERM_WINDOW,
        "st_s": SHORT_TERM_STEP,
        "mt_w": MID_TERM_WINDOW,
        "mt_s": MID_TERM_STEP
    }


    range_mt_max = int( round(max_mt - min_mt, 3) / step_mt)+1
    range_st_max = int( round(max_st - min_st, 3) / step_st)+1

    for mt in range(0,range_mt_max):
        print "\n\n\n\n"


        SHORT_TERM_WINDOW = min_st
        SHORT_TERM_STEP = round(SHORT_TERM_WINDOW*st_overl, 3)
        for st in range(0,range_st_max):

            # print_parameters(SHORT_TERM_WINDOW, SHORT_TERM_STEP, MID_TERM_WINDOW, MID_TERM_STEP, perc_train, use_svm, "aaa")

            if multiclassifier:
                accuracy = feature_and_train_multiclassifier(SAMPLES_PREFIX, SHORT_TERM_WINDOW, SHORT_TERM_STEP, MID_TERM_WINDOW,
                                                  MID_TERM_STEP, perc_train, confusion_matrix_perc, use_svm, VERBOSITY)
            else:
                accuracy = feature_and_train(SAMPLES_PREFIX, SHORT_TERM_WINDOW, SHORT_TERM_STEP, MID_TERM_WINDOW, MID_TERM_STEP,
                                 perc_train, confusion_matrix_perc, use_svm, VERBOSITY)

            SHORT_TERM_WINDOW = SHORT_TERM_WINDOW +step_st
            SHORT_TERM_STEP = round(SHORT_TERM_WINDOW*st_overl, 3)

            if accuracy > bestAcc:
                bestAcc = accuracy
                bestAccParams["st_w"] = SHORT_TERM_WINDOW
                bestAccParams["st_s"] = SHORT_TERM_STEP
                bestAccParams["mt_w"] = MID_TERM_WINDOW
                bestAccParams["mt_s"] = MID_TERM_STEP


        MID_TERM_WINDOW = MID_TERM_WINDOW+step_mt
        MID_TERM_STEP = round(MID_TERM_WINDOW*mt_overl, 3)


        print "\n\n\nMelhor precisao: {}".format(bestAcc)
        print "SHORT_TERM_WINDOW = {} \nSHORT_TERM_STEP = {} \nMID_TERM_WINDOW = {} \nMID_TERM_STEP = {}\n\n".format(
            bestAccParams["st_w"], bestAccParams["st_s"], bestAccParams["mt_w"], bestAccParams["mt_s"])


if __name__ == '__main__':

    PROJECT_PATH = "/mnt/c/dev/emotion_speech_recognition/" if IS_LINUX else "C:\dev\emotion_speech_recognition\\"

    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, choices=['portuguese', 'german'], help="Which audio DB to use. Options are: portuguese or german")
    parser.add_argument("--train", help="Train the machine.", action="store_true")
    parser.add_argument("--test", help="Test the model.", action="store_true")
    parser.add_argument("--testfile", help="File (with path) to test.")
    parser.add_argument("--multiclassifier", help="Use multiclassifier model.", action="store_true")

    # parser.add_argument("-p", action="store_true")

    args = parser.parse_args()

    print('\n===========================')
    print('Starting program...')
    print('===========================\n')


    # ============ PARAMETER DEFINITIONS ==============================================================================


    if args.db == "german":
        if args.multiclassifier:
            SAMPLES_PREFIX = PROJECT_PATH + 'audio_samples/german_emo_multiclassifier/'
        else:
            SAMPLES_PREFIX = PROJECT_PATH + 'audio_samples/german_emo/'
    else:
        if args.multiclassifier:
            SAMPLES_PREFIX = PROJECT_PATH + 'audio_samples/portuguese_multiclassifier/'
        else:
            SAMPLES_PREFIX = PROJECT_PATH + 'audio_samples/portuguese/'
    # ============ PARAMETER DEFINITIONS ==============================================================================

    # if args.p:
    #     parallel_method()
    #     SAMPLES_PREFIX = PROJECT_PATH + 'audio_samples/german_emo_all/'


    if args.db:

        if args.train:
            if args.test or args.testfile:
                parser.error("You can't use test or testfile when using train flag!")
            else:
                train_models(args.multiclassifier)

        elif args.test:
            if args.testfile:
                filee = args.testfile.split("/" if IS_LINUX else "\\")
                if len(filee) > 1:
                    filename = SAMPLES_PREFIX +("/" if IS_LINUX else "\\")+filee[-2]+("/" if IS_LINUX else "\\")+filee[-1]
                else:
                    filename = SAMPLES_PREFIX +("/" if IS_LINUX else "\\")+args.testfile

                # filename = SAMPLES_PREFIX + "A(fear)/03b02Aa.wav"
                test_file(filename)
            else:
                parser.error("You should pass the file path to test with flag --testfile /path/to/file/to/test")
        else:
            parser.error("You should specify to train or test with flags --train or --test")
    else:
        parser.error("You should specify which audio db to use with flag --db portuguese or --db german")



    # python cmd to split audios into smaller audios
    # python /mnt/c/dev/emotion_speech_recognition/src/dependencies/pyAudioAnalysis/audacityAnnotation2WAVs.py -d /mnt/c/dev/emotion_speech_recognition/audio_sample_to_split /mnt/c/dev/emotion_speech_recognition/audio_samples/

    print('\n===========================')
    print('Ending program...')
    print('===========================\n')
