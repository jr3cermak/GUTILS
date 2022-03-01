#!/usr/bin/env python3

# This program creates a catalog of the available *.dat files
# by start and end time.

import os, sys, datetime
import subprocess
import argparse
import glob

## Functions

def catalogFiles(dataDir, inputFile, catFile, showWarning):

    filesToProcess = glob.glob(os.path.join(dataDir, inputFile))
    catalog = {}

    for datFile in filesToProcess:
        if not(os.path.isfile(datFile)):
            sys.exit("Input file not found: %s" % (datFile))

        fp = open(datFile, 'r')
        hdrFlag = True
        numTags = 0
        numData = 0
        dataCols = []
        timeData = []
        presData = []

        while True:
            ln = fp.readline()
            if not(ln):
                break
            ln = ln.strip()
            if hdrFlag:
                # Try to determine the number of sensors; data columns
                if ln.find('sensors_per_cycle') >= 0:
                    data = ln.split()
                    numTags = int(data[1])
                # We assume when we run out of headers with a colon(:) we might be
                # reading into data.  We need to find the 'sci_m_present_time' and
                # 'sci_water_pressure' column for the data catalog.
                if ln.find(':') == -1:
                    data = ln.split()
                    # This starts the data section!
                    if len(data) == numTags:
                        hdrFlag = False
                        dataCols = data
                        sciTime = dataCols.index('sci_m_present_time')
                        sciPres = dataCols.index('sci_water_pressure')
            else:
                numData = numData + 1
                data = ln.split()
                if numData >= 3:
                    timeVal = data[sciTime]
                    if timeVal != 'NaN':
                        timeData.append(float(timeVal))
                    presVal = data[sciPres]
                    if presVal != 'NaN':
                        presData.append(float(presVal))

        fp.close()

        baseFile = os.path.basename(datFile)

        recIndex = None
        minTimeData = -9999.0
        maxTimeData = -9999.0
        minPresData = -9999.0
        maxPresData = -9999.0

        if len(timeData) > 0:
            minTimeData = min(timeData)
            recIndex = minTimeData
            maxTimeData = max(timeData)

        if len(presData) > 0:
            minPresData = min(presData)
            maxPresData = max(presData)

        if recIndex:
            catalog[recIndex] = {
                'file': baseFile,
                'minTime': minTimeData,
                'maxTime': maxTimeData,
                'minPres': minPresData,
                'maxPres': maxPresData
            }
        else:
            if showWarnings:
                print("WARNING: No date/time information for %s, skipping." % (baseFile))

    # Write out a catalog
    timeKeys = list(catalog.keys())
    timeKeys.sort()

    fp = open(catFile, 'w')
    fp.write("TbdFile, minTimeStr, maxTimeStr, minTime, maxTime, minPress, maxPress\n")
    fp.write("name, UTC, UTC, seconds, seconds, bars, bars\n")
    for timeKey in timeKeys:
        minTimeStr = datetime.datetime.utcfromtimestamp(catalog[timeKey]['minTime']).strftime('%Y-%m-%d %H:%M:%S')
        maxTimeStr = datetime.datetime.utcfromtimestamp(catalog[timeKey]['maxTime']).strftime('%Y-%m-%d %H:%M:%S')
        fp.write("%s, %s, %s, %f, %f, %f, %f\n" % (\
            catalog[timeKey]['file'],
            minTimeStr,
            maxTimeStr,
            catalog[timeKey]['minTime'],
            catalog[timeKey]['maxTime'],
            catalog[timeKey]['minPres'],
            catalog[timeKey]['maxPres']
        ))
    fp.close()

## Main program

# Add guard for sphinx
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataDir", type=str, help="directory with tbd files", required=True)
    parser.add_argument("--file", type=str, help="dat file or glob \*.dat", required=True)
    parser.add_argument("--catalog", type=str, help="output file name for catalog", required=True)
    parser.add_argument("--warn", help="warn about data files without date/time information",
        action="store_true", default=False)

    args = parser.parse_args()

    catFile = args.catalog
    dataDir = args.dataDir
    inputFile = args.file
    showWarnings = args.warn

    catalogFiles(dataDir, inputFile, catFile, showWarnings)
