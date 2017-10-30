#!/usr/bin/env python

import os
from collections import OrderedDict
from pyAudioAnalysis import audioTrainTest as aT
import argparse
import time
import sys
sys.path.append('C:\dev\emotion_speech_recognition\src\dependencies\pyAudioAnalysis')

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

def feature_and_train(samples_prefix, st_w, st_s, mt_w, mt_s, perc_train, confusion_matrix_perc, use_svm, verbosity, model_name):
    start = time.time()
    list_of_dirs_or_classes = []
    dirs = [d for d in os.listdir(samples_prefix) if os.path.isdir(os.path.join(samples_prefix, d))]
    for dire in dirs:
        list_of_dirs_or_classes.append(samples_prefix + str(dire))

    if use_svm:
        print "Starting training..."
        model_name = "models/SVM_"+model_name
        print_parameters(st_w, st_s, mt_w, mt_s, perc_train, use_svm, model_name)
        bestParam, bestAcc = aT.featureAndTrain(list_of_dirs_or_classes, mt_w, mt_s, st_w, st_s, "svm", model_name, False, perc_train, confusion_matrix_perc, verbosity=verbosity)
    else:
        print "Starting training..."

        model_name = "models/KNN_"+model_name

        print_parameters(st_w, st_s, mt_w, mt_s, perc_train, use_svm, model_name)
        bestParam, bestAcc = aT.featureAndTrain(list_of_dirs_or_classes, mt_w, mt_s, st_w, st_s, "knn", model_name, False, perc_train, confusion_matrix_perc, verbosity=verbosity)

    end = time.time()
    print "Finished in {:10.2f} seconds".format(end-start)
    return bestAcc

def test_model(prefix, model):
    dirs = os.listdir(prefix)

    use_svm = True
    model = ("models/SVM_" + model) if use_svm else ("models/KNN_" + model)

    print "\nTesting model: {}".format(model)
    print "Confusion Matrix:"
    for d in dirs:
        print "\t{}".format(d[:4]),
    print

    total_correct = 0
    total_files = 0
    for classs in dirs:
        files = os.listdir(prefix+classs)
        class_files_num = len(files)
        total_files = total_files+class_files_num
        classified = OrderedDict()
        for dir in dirs:
            classified[dir] = 0

        print "{}\t".format(classs[:4]),
        for file in files:
            file_to_test = prefix+classs+"/"+file
            class_classified = test_file(file_to_test, model, use_svm)
            classified[class_classified] = classified[class_classified]+1
            if class_classified == classs:
                total_correct = total_correct+1

        for key in classified.keys():
            print "{}\t".format( round((classified[key]/float(class_files_num))*100,2) ),
        print
    print

    print "General precision is {}".format( round(total_correct/float(total_files)*100,2) )

def test_file(filename_to_test, model_name, use_svm=True):
    if os.path.isfile(filename_to_test):
        start = time.time()
        if use_svm:
            r, P, classNames = aT.fileClassification(filename_to_test, model_name, "svm")
        else:
            r, P, classNames = aT.fileClassification(filename_to_test, model_name, "svm")

        chosen = 0.0
        chosenClass = ""

        if len(P) == len(classNames):
            for i in range(0, len(P), 1):

                if P[i] > chosen:
                    chosen = P[i]
                    chosenClass = classNames[i]

        end = time.time()
        # DEBUG
        # print "\n\nThe audio file was classified as {} with prob {}% in {:10.2f} seconds\n\n".format(chosenClass, round(chosen*100, 2), end - start )
        return chosenClass
    else:
        print "File doesnt exists: {}".format(filename_to_test)
        return None

def train_until_get_better_acc(samples_prefix, model_name):

    best_acc = 0
    for i in range(0,1000):
        accuracy = train_SVM(samples_prefix, model_name)
        print "\nCurrent accuracy: {}".format(accuracy)
        print "Best accuracy: {}\n\n".format(best_acc)

        if accuracy > best_acc:
            prefix = samples_prefix+"../../src/models/"
            os.rename(prefix+"SVM_port_single", prefix+"best_SVM_port_single")
            os.rename(prefix+"SVM_port_single.arff", prefix+"best_SVM_port_single.arff")
            os.rename(prefix+"SVM_port_singleMEANS", prefix+"best_SVM_port_singleMEANS")
            best_acc = accuracy

        if accuracy >= 84.6:
            break

def train_SVM(samples_prefix, model_name):

    # ======= PORTUGUESE BEST CONFIGURATION ========
    SHORT_TERM_WINDOW = 0.1
    SHORT_TERM_STEP = 0.033
    MID_TERM_WINDOW = 1.2
    MID_TERM_STEP = 0.6
    # ======= PORTUGUESE BEST CONFIGURATION ========

    # ======= GERMAN BEST CONFIGURATION ========
    SHORT_TERM_WINDOW = 0.036
    SHORT_TERM_STEP = 0.012
    MID_TERM_WINDOW = 1.3
    MID_TERM_STEP = 0.65
    # ======= GERMAN BEST CONFIGURATION ========

    confusion_matrix_perc = True
    use_svm = True
    perc_train = 0.75
    VERBOSITY = False

    return feature_and_train(samples_prefix, SHORT_TERM_WINDOW, SHORT_TERM_STEP, MID_TERM_WINDOW, MID_TERM_STEP,
                                 perc_train, confusion_matrix_perc, use_svm, VERBOSITY, model_name)

def train_KNN(samples_prefix, model_name):
    SHORT_TERM_WINDOW = 0.036
    SHORT_TERM_STEP = 0.012
    MID_TERM_WINDOW = 1.3
    MID_TERM_STEP = 0.65

    confusion_matrix_perc = True
    use_svm = False
    perc_train = 0.75
    VERBOSITY = False

    feature_and_train(samples_prefix, SHORT_TERM_WINDOW, SHORT_TERM_STEP, MID_TERM_WINDOW, MID_TERM_STEP,
                                 perc_train, confusion_matrix_perc, use_svm, VERBOSITY, model_name)

def brute_force_training(samples_prefix):
    min_st   = 0.020
    max_st   = 0.100
    step_st  = 0.001
    st_overl = 0.33

    min_mt   = 1.000
    max_mt   = 3.000
    step_mt  = 0.100
    mt_overl = 0.5


    MID_TERM_WINDOW = min_mt
    MID_TERM_STEP = round(MID_TERM_WINDOW*mt_overl, 3)
    SHORT_TERM_WINDOW = min_st
    SHORT_TERM_STEP = round(SHORT_TERM_WINDOW*st_overl, 3)

    confusion_matrix_perc = True
    use_svm = True
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

            # print_parameters(SHORT_TERM_WINDOW, SHORT_TERM_STEP, MID_TERM_WINDOW, MID_TERM_STEP, perc_train, USE_SVM, "aaa")

            accuracy = feature_and_train(samples_prefix, SHORT_TERM_WINDOW, SHORT_TERM_STEP, MID_TERM_WINDOW, MID_TERM_STEP,
                                 perc_train, confusion_matrix_perc, use_svm, VERBOSITY, "aaaaaaaa")

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

    PROJECT_PATH = "/mnt/c/dev/emotion_speech_recognition/"

    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, choices=['portuguese', 'german'], help="Which audio DB to use. Options are: portuguese or german")
    parser.add_argument("--train", help="Train the machine.", action="store_true")
    parser.add_argument("--test", help="Test the model.", action="store_true")

    args = parser.parse_args()

    print('\n===========================')
    print('Starting program...')
    print('===========================\n')


    MODEL_NAME = ""
    if args.db == "german":
        SAMPLES_PREFIX = PROJECT_PATH + 'audio_samples/german_emo/'
        TEST_PREFIX = PROJECT_PATH + 'audio_samples/german_emo_test/'
        MODEL_NAME = "german_single"
    else:
        SAMPLES_PREFIX = PROJECT_PATH + 'audio_samples/portuguese/'
        TEST_PREFIX = PROJECT_PATH + 'audio_samples/portuguese_test/'
        MODEL_NAME = "port_single"

    if args.db:
        if args.train:
            if args.test:
                parser.error("You can't use test and train flag at the same time!")
            else:
                #brute_force_training(SAMPLES_PREFIX)
                # train_until_get_better_acc(SAMPLES_PREFIX, MODEL_NAME)
                train_SVM(SAMPLES_PREFIX, MODEL_NAME)
                # train_KNN(SAMPLES_PREFIX, MODEL_NAME)

        elif args.test:
            test_model(TEST_PREFIX, MODEL_NAME)

        else:
            parser.error("You should specify to train or test with flags --train or --test")
    else:
        parser.error("You should specify which audio db to use with flag --db portuguese or --db german")



    # python cmd to split audios into smaller audios
    # python /mnt/c/dev/emotion_speech_recognition/src/dependencies/pyAudioAnalysis/audacityAnnotation2WAVs.py -d /mnt/c/dev/emotion_speech_recognition/audio_sample_to_split /mnt/c/dev/emotion_speech_recognition/audio_samples/




    print('\n===========================')
    print('Ending program...')
    print('===========================\n')
