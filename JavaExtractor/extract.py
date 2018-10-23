#!/usr/bin/python

import multiprocessing
import os
import sys
import shutil
import subprocess
from threading import Timer
import sys
from argparse import ArgumentParser
from subprocess import Popen, PIPE, STDOUT, call



def get_immediate_subdirectories(a_dir):
    return [(os.path.join(a_dir, name)) for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


TMP_DIR = ""

def ParallelExtractDir(dir):
    ExtractFeaturesForDir(dir, "")


def ExtractFeaturesForDir(args, dir, prefix):
    command = ['java', '-cp', args.jar, 'JavaExtractor.App',
               '--max_path_length', str(args.max_path_length), '--max_path_width', str(args.max_path_width),
               '--dir', dir, '--num_threads', str(args.num_threads)]

    # print command
    # os.system(command)
    kill = lambda process: process.kill()
    outputFileName = TMP_DIR + prefix + dir.split('/')[-1]
    failed = False
    with open(outputFileName, 'a') as outputFile:
        sleeper = subprocess.Popen(command, stdout=outputFile, stderr=subprocess.PIPE)
        timer = Timer(600000, kill, [sleeper])

        try:
            timer.start()
            stdout, stderr = sleeper.communicate()
        finally:
            timer.cancel()

        if sleeper.poll() == 0:
            if len(stderr) > 0:
                print(sys.stderr, stderr, file=sys.stdout)
        else:
            print(sys.stderr, 'dir: ' + str(dir) + ' was not completed in time', file=sys.stdout)
            failed = True
            subdirs = get_immediate_subdirectories(dir)
            for subdir in subdirs:
                ExtractFeaturesForDir(args, subdir, prefix + dir.split('/')[-1] + '_')
    if failed:
        if os.path.exists(outputFileName):
            os.remove(outputFileName)


def ExtractFeaturesForDirsList(args, dirs):
    global TMP_DIR
    TMP_DIR = "./tmp/feature_extractor%d/" % (os.getpid())
    if os.path.exists(TMP_DIR):
        shutil.rmtree(TMP_DIR, ignore_errors=True)
    os.makedirs(TMP_DIR)
    try:
        # p = multiprocessing.Pool(4)
        # p.map(ParallelExtractDir, dirs)
        for dir in dirs:
            ExtractFeaturesForDir(args, dir, '')
        output_files = os.listdir(TMP_DIR)
        for f in output_files:
            os.system("cat %s/%s" % (TMP_DIR, f))
    finally:
        shutil.rmtree(TMP_DIR, ignore_errors=True)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-maxlen", "--max_path_length", dest="max_path_length", required=False, default=8)
    parser.add_argument("-maxwidth", "--max_path_width", dest="max_path_width", required=False, default=2)
    parser.add_argument("-threads", "--num_threads", dest="num_threads", required=False, default=64)
    parser.add_argument("-j", "--jar", dest="jar", required=True)
    parser.add_argument("-dir", "--dir", dest="dir", required=False)
    parser.add_argument("-file", "--file", dest="file", required=False)
    args = parser.parse_args()

    if args.file is not None:
        command = 'java -cp ' + args.jar + ' JavaExtractor.App --max_path_length ' + \
                  str(args.max_path_length) + ' --max_path_width ' + str(args.max_path_width) + ' --file ' + args.file
        os.system(command)
    elif args.dir is not None:
        subdirs = get_immediate_subdirectories(args.dir)
        ExtractFeaturesForDirsList(args, subdirs)
    
