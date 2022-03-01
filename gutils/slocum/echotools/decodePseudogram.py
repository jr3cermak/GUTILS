#!/usr/bin/env python3

'''
This script contains functions that carries out the process
of obtaining the pseudogram data array in either CSV (ASCII,
spreadsheet) or image(PNG) form.  The output can be directed
to a file or standard output (stdout).

Data written as output is in three columns (comma separated
values): Timestamp, Depth, Density.  A metadata desciption
is given below.

Image output is controlled by the filename extension provided.
A typical format is PNG.  Use ".png" in the filename to produce
a PNG formatted image.

Metadata
--------
    Here is the metadata description for each of the coloums
    of this dataset (ASCII, spreadsheet):

    +--------------+--------------------------------------------------+
    | Column       | Description                                      |
    +--------------+--------------------------------------------------+
    | Timestamp    | seconds since 01-01-1970 epoch; timezone GMT/UTC |
    +--------------+--------------------------------------------------+
    | Depth        | depth (meters)                                   |
    +--------------+--------------------------------------------------+
    | Density      | Sv (dB)                                          |
    +--------------+--------------------------------------------------+

Command line arguments
----------------------

::

    -h, --help            show this help message and exit
    --tbdFile TBDFILE     full or relative path with filename
                          to Teledyne glider tbd binary input file
    --sbdFile SBDFILE     full or relative path with filename
                          to Teledyne glider sbd binary input file
    --cacheDir CACHEDIR   Directory with glider cache files;
                          default current directory
    --dbd2asc DBD2ASC     full or relative path with filename
                          to glider dbd2asc binary
    --csvOut CSVOUT       full or realative path with filename
                          to write CSV output or 'stdout'; default None
    --csvHeader           (flag) include header with CSV output; default False
    --imageOut IMAGEOUT   filename to write image or stdout; default None
    --debug               (flag) show extra debugging for this
                          python script; default False
    --echosounderRange ECHOSOUNDERRANGE
                          Echosounder range; default -60.0 (meters)
                          instrument facing up; positive values
                          instrument facing down
    --title               optional figure title; default None
    --useScatterPlot      (flag) Plot using a scatterplot
                          instead of a time/depth binned
                          (raster image) plot; default False
    --binnedDepthLabels   (flag) Use the original depth bin
                          labels instead of the adjusted depth
                          labels for the time/depth binned plot;
                          default False

'''

import struct
import os, sys, datetime
import argparse
import numpy as np

# Teledyne Webb generic python classes
# and functions
import teledyne

# Functions

def printAttributes(myObj):
    '''
    This is a convienence function for printing object attribute
    information for the supplied object.  This function writes
    to standard output and does not print any objects with
    an underscore prefix.  This was mainly used for debugging
    during development.

    Parameters
    ----------
    myObj : :obj:`any`, required

    Returns
    -------
    Prints information to standard output.
    '''

    for att in dir(myObj):
        # Skip __ attributes
        if att.startswith('_'):
            continue
        print("  %s: %s" % (att, getattr(myObj, att)))

def showHelp(parser):
    '''
    This prints the program description and arguments to standard ouptut and exits.
    '''
    parser.print_help()
    sys.exit()

# Main Program

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='''
Pseudogram decoder: This reads a Teledyne Web glider tbd file that has encoded
pseudograms.  In some cases, additional metadata is required from the sbd
glider file.   To generate output, --csvOut and/or --imageOut must specify full
or relative path with a filename or stdout.  The arguments --csvOut and
--imageOut cannot both be stdout.  The image format is controlled by file
extention (png, jpg, pdf, gif).  The default echosounderRange is -60.0 meters
for an echosounder instrument points towards the surface.  If the instrument is
pointing down, use a positive value (range) in meters.

''')

    parser.add_argument("--tbdFile", help="full or relative path with filename to glider tbd binary input file", type=str)
    parser.add_argument("--sbdFile", help="full or relative path with filename to glider sbd binary input file", type=str)
    parser.add_argument("--cacheDir", help="Directory with glider cache files; default current directory", type=str)
    parser.add_argument("--dbd2asc", help="full or relative path with filename to glider dbd2asc binary", type=str)
    parser.add_argument("--csvOut", help="full or realative path with filename to write CSV output or 'stdout'; default None", type=str, default=None)
    parser.add_argument("--csvHeader", help="(flag) include header with CSV output; default False", action="store_true", default=False)
    parser.add_argument("--imageOut", help="filename to write image or stdout; default None", type=str, default=None)
    parser.add_argument("--debug", help="(flag) show extra debugging for this python script; default False", action="store_true", default=False)
    parser.add_argument("--echosounderRange",
            help="Echosounder range; default -60.0 (meters) instrument facing up; positive values instrument facing down", default=-60.0)
    parser.add_argument("--useScatterPlot", help="(flag) Plot using a scatterplot instead of a time/depth binned (raster image) plot; default False", action="store_true", default=False)
    parser.add_argument("--binnedDepthLabels", help="(flag) Use the original depth bin labels instead of the adjusted depth labels for the time/depth binned plot; default False", action="store_true", default=False)
    parser.add_argument("--title", help="optional figure title; default None", type=str)

    args = parser.parse_args()
    debugFlag = args.debug

    # Initial arguments
    if debugFlag:
        print("Initial arguments:")
        printAttributes(args)

    # csv and image cannot be both stdout
    if args.csvOut == 'stdout' and args.imageOut == 'stdout':
        print("ERROR: The csvOut and imageOut options cannot both be set to 'stdout'.")
        print("       Run the decoder command separately for csvOut and imageOut using 'stdout'.")
        showHelp(parser)

    # Check for glider tbd file
    if args.tbdFile is None:
        print("ERROR: A glider tbd file needs to be specified.\n")
        showHelp(parser)

    if not(os.path.isfile(args.tbdFile)):
        print("ERROR: Glider tbd file not found: %s\n" % (args.tbdFile))
        showHelp(parser)

    if args.sbdFile:
        if not(os.path.isfile(args.sbdFile)):
            print("ERROR: Glider sbd file not found: %s\n" % (args.sbdFile))
            showHelp(parser)

    # Determine glider cache file directory
    if args.cacheDir is None:
        args.cacheDir = os.getcwd()
    else:
       if not(os.path.isdir(args.cacheDir)):
           print("ERROR: Glider cache directory not found: %s\n" % (args.cacheDir))
           showHelp(parser)

    # Make sure the Teledyne Webb linux binary is present
    if args.dbd2asc is None:
        print("ERROR: Teledyne Webb linux binary needs to be specified.\n")
        showHelp(parser)
    if not(os.path.isfile(args.dbd2asc)):
        print("ERROR: Teledyne Webb linux binary not found: %s\n" % (args.dbd2asc))
        showHelp(parser)

    # Final arguments
    if debugFlag:
        print("Processed arguments:")
        printAttributes(args)

    # This will create a class variable "data" which is a
    # dictionary with the following keys:
    #    asc: array of data elements as seen from dbd2asc
    #    columns: list of columns
    #    byteSize: list of byte sizes
    #    units: list of units
    #    cacheMetadata: dict()
    #        sensorCount: cache file "sensor_per_cycle"
    #        factored: cache file "sensor_list_factored"
    #        stByteNum: cache file "state_bytes_per_cycle"
    #        totalSensors: cache file "total_num_sensors"
    #        cacheFile: cache file "sensor_list_crc"
    glider = teledyne.Glider(
        tbdFile = args.tbdFile,
        sbdFile = args.sbdFile,
        cacheDir = args.cacheDir,
        dbd2asc = args.dbd2asc,
        debugFlag = debugFlag
    )

    # Read tbd file
    glider.readTbd()

    # Read sbd file (if available)
    if args.sbdFile:
        glider.readSbd()

    # Now that we have the appropriate cache file metadata, we
    # can now read the pseudogram from the binary tbd file.
    glider.readPseudogram()

    # Code from echoGenNew.py in teledyne.createPseudogramSpreadsheet
    glider.createPseudogramSpreadsheet(args)

    # Handle spreadsheet: send to stdout or write to file
    if args.csvOut:
        glider.handleSpreadsheet(args)

    # Handle image: send to stdout or write to file
    if args.imageOut:
        glider.handleImage(args)
