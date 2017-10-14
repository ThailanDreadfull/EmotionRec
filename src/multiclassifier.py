#!/usr/bin/env python

import os
import shutil

def create_multiclassifier_folders():

    # ========= GERMAN ============
    PREFIX = '/mnt/c/dev/emotion_speech_recognition/audio_samples/german_emo_multi_base/'
    PREFIX_MULTI = '/mnt/c/dev/emotion_speech_recognition/audio_samples/german_emo_MULTI/'
    # ========= GERMAN ============

    # ========= PORTUGUESE ============
    # PREFIX = '/mnt/c/dev/emotion_speech_recognition/audio_samples/portuguese/'
    # PREFIX_MULTI = '/mnt/c/dev/emotion_speech_recognition/audio_samples/portuguese_MULTI/'
    # ========= PORTUGUESE ============

    if not os.path.exists(PREFIX_MULTI):
        print "Creating dir {}".format(PREFIX_MULTI)
        os.makedirs(PREFIX_MULTI)

    dir_classes = [x[0] for x in os.walk(PREFIX)]
    dir_classes = dir_classes[1:]
    for dir in dir_classes:
        class_name = dir.split('/')[-1]

        if not os.path.exists(PREFIX_MULTI+class_name):
            print "Creating dir {}".format(PREFIX_MULTI+class_name)
            os.makedirs(PREFIX_MULTI+class_name)
        if not os.path.exists(PREFIX_MULTI + class_name+'/'+ class_name):
            print "Creating dir {}".format(PREFIX_MULTI + class_name+'/'+ class_name)
            os.makedirs(PREFIX_MULTI + class_name+'/'+ class_name)
        if not os.path.exists(PREFIX_MULTI + class_name + '/all'):
            print "Creating dir {}".format(PREFIX_MULTI + class_name + '/all')
            os.makedirs(PREFIX_MULTI + class_name + '/all')


        #removing all copies
        onlyfiles = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
        for file in onlyfiles:
            if "Copia" in file:
                os.remove(dir+'/'+file)


        onlyfiles = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
        for file in onlyfiles:
            print "Copying file from {} to {}".format(dir+'/'+file, PREFIX_MULTI+class_name+'/'+class_name+'/'+file)
            shutil.copy2(dir+'/'+file, PREFIX_MULTI+class_name+'/'+class_name+'/'+file)


        for class_dir in dir_classes:
            c_name = class_dir.split('/')[-1]
            if c_name == class_name:
                continue

            only_files = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]
            for file in only_files:
                print "--Copying file from {} to {}".format(class_dir + '/' + file, PREFIX_MULTI + class_name + '/all/' + file)
                shutil.copy2(class_dir + '/' + file, PREFIX_MULTI + class_name + '/all/' + file)


if __name__ == '__main__':
    create_multiclassifier_folders()

