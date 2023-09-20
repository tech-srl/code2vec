#!/usr/bin/python

import itertools
import multiprocessing
import os
import shutil
import subprocess
from threading import Timer
from argparse import ArgumentParser

def get_immediate_subdirectories(a_dir):
    return [(os.path.join(a_dir, name)) for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

def ParallelExtractDir(args, tmpdir, dir_):
    ExtractFeaturesForDir(args, tmpdir, dir_, "")

def ExtractFeaturesForDir(args, tmpdir, dir_, prefix):
    command = ['java', '-cp', args.jar, 'JavaExtractor.App',
               '--max_path_length', str(args.max_path_length), '--max_path_width', str(args.max_path_width),
               '--dir', dir_, '--num_threads', str(args.num_threads)]
    kill = lambda process: process.kill()
    outputFileName = tmpdir + prefix + dir_.split('/')[-1]
    failed = False
    with open(outputFileName, 'a') as outputFile:
        sleeper = subprocess.Popen(command, stdout=outputFile, stderr=subprocess.PIPE,)
        timer = Timer(600000, kill, [sleeper])
        try:
            timer.start()
            stdout, stderr = sleeper.communicate()
        finally:
            timer.cancel()

        if sleeper.poll() != 0:
            failed = True
            subdirs = get_immediate_subdirectories(dir_)
            for subdir in subdirs:
                ExtractFeaturesForDir(args, subdir, prefix + dir_.split('/')[-1] + '_')
                
    if failed and os.path.exists(outputFileName):
        os.remove(outputFileName)

def ExtractFeaturesForDirsList(args, dirs):
    tmp_dir = f"./tmp/feature_extractor{os.getpid()}/"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir)
    
    for i in range(0, len(dirs), args.batch_size):
        batch_dirs = dirs[i:i + args.batch_size]
        timeout_seconds = 60  # timeout setting
        try:
            with multiprocessing.Pool(4) as p:
                result = p.starmap_async(ParallelExtractDir, zip(itertools.repeat(args), itertools.repeat(tmp_dir), batch_dirs))
                result.get(timeout=timeout_seconds)
        except multiprocessing.TimeoutError:
            continue

        output_files = os.listdir(tmp_dir)
        for f in output_files:
            os.system("cat %s/%s" % (tmp_dir, f))
            os.remove(os.path.join(tmp_dir, f))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-maxlen", "--max_path_length", dest="max_path_length", required=False, default=8)
    parser.add_argument("-maxwidth", "--max_path_width", dest="max_path_width", required=False, default=2)
    parser.add_argument("-threads", "--num_threads", dest="num_threads", required=False, default=64)
    parser.add_argument("-j", "--jar", dest="jar", required=True)
    parser.add_argument("-dir", "--dir", dest="dir", required=False)
    parser.add_argument("-file", "--file", dest="file", required=False)
    #  add a new parameter batch_size
    parser.add_argument("-batch_size", "--batch_size", dest="batch_size", required=False, default=3, type=int)

    args = parser.parse_args()

    if args.file is not None:
        command = 'java -cp ' + args.jar + ' JavaExtractor.App --max_path_length ' + \
                  str(args.max_path_length) + ' --max_path_width ' + str(args.max_path_width) + ' --file ' + args.file
        os.system(command)
    elif args.dir is not None:
        subdirs = get_immediate_subdirectories(args.dir)
        to_extract = subdirs if subdirs else [args.dir.rstrip('/')]
        ExtractFeaturesForDirsList(args, to_extract)
