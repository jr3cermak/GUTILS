#!/usr/bin/env python3

import os, sys
import subprocess
import argparse
import glob

## Functions

def processTbdFile(tbdFile, cacheDir, binDir, dataDir, renameFlag):

    filesToProcess = glob.glob(os.path.join(dataDir, tbdFile))
    #print("dataDir", dataDir)
    #print("tbdFile", tbdFile)
    #print("Files to process:", filesToProcess)

    for tbdFile in filesToProcess:

        # Check to see if file exists
        fullFile = tbdFile
        if not(os.path.isfile(fullFile)):
            sys.exit("TBD file does not exist: %s" % (fullFile))

        # Rename the file on the fly if requested
        if tbdFile.find('unit_507') >= 0 and renameFlag:
            newFile = tbdFile.replace('unit_507', 'uaf-gretel')
            os.rename(fullFile, newFile)
            fullFile = newFile

        # Set output file
        outputFile = "%s.dat" % (fullFile[:-4])

        # Check for executable dbd2asc
        dbdExec = os.path.join(binDir, 'dbd2asc')
        if not(os.path.isfile(dbdExec)):
            sys.exit("Executable not found: %s" % (dbdExec))

        # Decode tbd file and store output
        cmd = [dbdExec, '-c', cacheDir, fullFile]
        processOutput = subprocess.run(cmd, stdout=subprocess.PIPE)
        returnCode = processOutput.returncode

        if returnCode != 0:
            sys.exit("dbd2asc failed to run: %s" % (cmd))

        #breakpoint()
        fp = open(outputFile, 'wb')
        fp.write(processOutput.stdout)
        fp.close()

## Main program

# Add guard for sphinx
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--binDir", type=str, help="directory with teledyne executables", required=True)
    parser.add_argument("--dataDir", type=str, help="directory with tbd files", required=True)
    parser.add_argument("--cacheDir", type=str, help="directory with cache files", required=True)
    parser.add_argument("--file", type=str, help="tbd file", required=True)
    parser.add_argument("--rename", action="store_true", help="rename unit_507 to uaf-gretel")

    args = parser.parse_args()

    renameFlag = args.rename
    binDir = args.binDir
    dataDir = args.dataDir
    tbdFile = args.file
    cacheDir = args.cacheDir

    processTbdFile(tbdFile, cacheDir, binDir, dataDir, renameFlag)
