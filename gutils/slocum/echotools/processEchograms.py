#!/usr/bin/env python3

'''
This script contains functions that decode an echogram from
the echometrics data stream (if present). The resultant
information can be:

 - Rendered in plot a plot: binned, scatter or pcolormesh
 - Saved to a CSV file
 - Saved to a netCDF file
 - Available as an xarray object
 - Available as a pandas object

The output can be directed to a file or standard output (stdout).
The netCDF may only be saved to a file.  The netCDF variables may
be saved as separate files for ease of aggregation.

Data written as CSV output is in three columns (comma separated
values): Timestamp, Depth, Density.  A metadata desciption
is given below.

Metadata for the netCDF file is pulled from deployment configuration
files: deployment.json and instruments.json; A general template file
is also required for generic metadata.

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
    --inpDir INPDIR       full or relative path to directory with
                          glider data
    --inpFile INPFILE     glider input file; one segment
    --t T                 data type: sfmc, rt, delayed
    --tbdFile TBDFILE     full or relative path with filename
                          to Teledyne glider tbd binary input file
    --sbdFile SBDFILE     full or relative path with filename
                          to Teledyne glider sbd binary input file
    --cacheDir CACHEDIR   Directory with glider cache files;
                          default current directory
    --dbd2asc DBD2ASC     full or relative path with filename
                          to glider dbd2asc binary
    --csvOut CSVOUT       full or relative path with filename
                          to write CSV output or 'stdout'; default None
    --ncDir NCDIR         full or relative path of a directory for
                          writing netCDF file(s); default None
    --ncFormat NCFORMAT   alternate time format for saving netCDF files
    --csvHeader           (flag) include header with CSV output; default False
    --imageOut IMAGEOUT   filename to write image or stdout; default None
    --outDir OUTDIR       output directory for csv and plot files; default None
    --debug               (flag) show extra debugging for this
                          python script; default False
    --echogramBins ECHOSOUNDERRANGE
                          Number of bins in use by the echogram;
                          default 20
    --echogramRange ECHOSOUNDERRANGE
                          Echogram range; default -60.0 (meters)
                          instrument facing up; positive values
                          instrument facing down
    --title               optional figure title; default None
    --plotType            One or more of binned, scatter, pcolormesh or
                          all; default binned
    --binnedDepthLabels   (flag) Use the original depth bin
                          labels instead of the adjusted depth
                          labels for the time/depth binned plot;
                          default False
    --deploymentDir       full or relative path to deployment and
                          instrument json files; default None
    --templateDir         full or relative path to glider
                          template directory; default None
    --saveBits            output the echogram bits for
                          validation.
'''

import os, sys, datetime, glob, json
import argparse

# Teledyne Webb generic python classes
# and functions

try:
    # Try to import for GUTILS first
    import teledyne
except:
    # This is the conventional method for echotools
    from echotools.parsers.slocum import teledyne

# Functions

def printArgs(args):
    '''
    This is a convienence function for printing dictionary
    elements.

    Parameters
    ----------
    args : :obj:`dict()`, required

    Returns
    -------
    Prints information to standard output.
    '''

    for key in args.keys():
        print("  %s: %s" % (key, args[key]))


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
    This prints the program description and arguments to standard output and exits.
    '''
    parser.print_help()
    sys.exit()

def sussFileType(args):
    '''
    Suss out file types we are attempting to work with based on the output directory path.
    '''

    print("ARGS:",args)

    return None

# Main Program

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='''
Echogram processing: This reads Teledyne Web glider files. The glider files
may include echograms from an echosounder. This program can operate
on a single file or a whole deployment of files.
''')

    parser.add_argument("--inpDir", help="full or relative path to glider input files", type=str, default=None)
    parser.add_argument("--inpFile", help="glider input file(s)", type=str, default=None)
    parser.add_argument("-t", help="glider file type: sfmc, rt or delayed", type=str, default=None)
    parser.add_argument("--tbdFile", help="full or relative path with filename to glider tbd binary input file", type=str, default=None)
    parser.add_argument("--sbdFile", help="full or relative path with filename to glider sbd binary input file", type=str, default=None)
    parser.add_argument("--cacheDir", help="Directory with glider cache files; default current directory", type=str, default=None)
    parser.add_argument("--dbd2asc", help="full or relative path with filename to glider dbd2asc binary", type=str, default=None)
    parser.add_argument("--csvOut", help="full or relative path with filename to write CSV output or 'stdout'; default None", type=str, default=None)
    parser.add_argument("--csvHeader", help="(flag) include header with CSV output; default False", action="store_true", default=False)
    parser.add_argument("--ncDir", help="full or relative path with filename to write netCDF output; default None", type=str, default=None)
    parser.add_argument("--ncSeparate", help="save tbd, sbd and sv data separately; default False", action="store_true", default=False)
    parser.add_argument("--imageOut", help="filename to write image or stdout; default None", type=str, default=None)
    parser.add_argument("--outDir", help="output directory for csv and plot files", type=str, default=None)
    parser.add_argument("--debug", help="(flag) show extra debugging for this python script; default False", action="store_true", default=False)
    parser.add_argument("--echogramBins",
            help="Echogram bins; default 20", type=int, default=20)
    parser.add_argument("--echogramRange",
            help="Echogram range; default -60.0 (meters) instrument facing up; positive values instrument facing down",
            type=float, default=-60.0)
    parser.add_argument("--plotType", help="Use one or more of binned, scatter, pcolormesh or all; default binned", type=str, default="binned")
    parser.add_argument("--binnedDepthLabels", help="(flag) Use the original depth bin labels instead of the adjusted depth labels for the time/depth binned plot; default False", action="store_true", default=False)
    parser.add_argument("--title", help="optional figure title; default None", type=str)
    parser.add_argument("--deploymentDir", help="full or relative path directory with json files", type=str, default=None)
    parser.add_argument("--template", help="full or relative path to metadata template", type=str, default=None)
    parser.add_argument("--templateDir", help="full or relative path to metadata template directory", type=str, default=None)
    parser.add_argument("--dacOverlay", help="full or relative path to an overlay configuration file", type=str, default=None)
    parser.add_argument("--saveBits", help="full or relative path to save the bit encoding of the echogram", type=str, default=None)
    args = parser.parse_args()
    args = vars(args)

    # Set some initial defaults
    args['echogramRangeUnits'] = 'meters'
    debugFlag = args.get('debug')

    # Initial arguments
    if debugFlag:
        print("Initial arguments:")
        printArgs(args)

    file_type = args.get('t')
    allowed_file_types = ['sfmc', 'rt', 'delayed']
    if not(file_type in allowed_file_types):
        file_type = sussFileType(args)
        if not(file_type in allowed_file_types):
            print("Allowed file types (-t):", allowed_file_types)
            sys.exit()

    # Set the file and ext mask to use
    file_mask = None
    ext_mask = None
    if file_type in ['sfmc', 'rt']:
        file_mask = '*.[stST][bB][dD]'
        ext_mask = '.[stST][bB][dD]'
    else:
        file_mask = '*.[deDE][bB][dD]'
        ext_mask = '.[deDE][bB][dD]'
    args['ext_mask'] = ext_mask
    args['file_mask'] = file_mask

    # csv and image cannot be both stdout
    if args['csvOut'] == 'stdout' and args['imageOut'] == 'stdout':
        print("ERROR: The csvOut and imageOut options cannot both be set to 'stdout'.")
        print("       Run the decoder command separately for csvOut and imageOut using 'stdout'.")
        showHelp(parser)

    '''
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
    '''

    # Determine glider cache file directory
    if args['cacheDir'] is None:
        args['cacheDir'] = os.getcwd()
    else:
       if not(os.path.isdir(args['cacheDir'])):
           print("ERROR: Glider cache directory not found: %s\n" % (args['cacheDir']))
           showHelp(parser)

    # Make sure the Teledyne Webb linux binary is present
    ''' deprecated
    if args.dbd2asc is None:
        print("ERROR: Teledyne Webb linux binary needs to be specified.\n")
        showHelp(parser)
    if not(os.path.isfile(args.dbd2asc)):
        print("ERROR: Teledyne Webb linux binary not found: %s\n" % (args.dbd2asc))
        showHelp(parser)
    '''

    # Scan for input files or operate on a "binary" directory of
    # files like GUTILS!

    # This can be a single file or multiple files by file glob and
    # be relative to args.inpDir.  If only an input directory is
    # specified, assume we were pointed at a "binary" directory
    # to process.
    fileGroups = []
    inputFiles = []

    if args['inpFile']:
        if args['inpDir']:
            # Specify directory and filename
            inputFiles = glob.glob(os.path.join(args['inpDir'], "%s%s" % (args['inpFile'], ext_mask)))
            if len(inputFiles) > 0:
                fileGroups.append(inputFiles)
        else:
            # Only a filename is specified
            inputFiles = glob.glob(os.path.join("%s%s" % (args['inpFile'], ext_mask)))
            if len(inputFiles) > 0:
                fileGroups.append(inputFiles)
    else:
        # Only a directory path is supplied, assume this is a directory
        # full of files to process
        if args['inpDir']:
            inputFiles = glob.glob(os.path.join(args['inpDir'], file_mask))
            glider = teledyne.Glider()
            fileGroups = list(glider.groupFiles(inputFiles).values())

    if len(fileGroups) == 0:
        print("ERROR: No input files were specified.")
        sys.exit()

    # Loop through each file group
    for inputFiles in fileGroups:

        # This will create a class variable "data" which is a
        # dictionary with the following keys that match the
        # file extension of the DBD file.
        glider = teledyne.Glider()

        # Pass the debugFlag immediately
        glider.debugFlag = debugFlag

        # Pass all the program arguments to the glider object
        glider.args = args

        # At this point, load any specified deployment metadata (if specified)

        # If ncDir is specified, then --deploymentDir and --templateDir must be
        # specified.

        # If writing netCDF files, the following information is needed
        # Deployment files: deployment.json, instrument.json
        # Metadata/DAC file: slocum_dac.json or some other template
        # Auxillary configuration/DAC file: in the same path as the deployment directory
        glider.loadMetadata()
        args = glider.updateArgumentsFromMetadata()
        glider.calculateMissionPlan()

        # Final arguments
        if debugFlag:
            print("Processed arguments:")
            printArgs(glider.args)

        # Process DBD files
        for inputFile in inputFiles:
            if debugFlag:
                print("Processing: %s" % (inputFile))
            glider.readDbd(inputFile=inputFile, cacheDir=args['cacheDir'])

        # Now that we have the appropriate cache file metadata, we
        # can now read the echogram from tbd data (if available)
        glider.readEchogram()

        # Code from echoGenNew.py in teledyne.createPseudogramSpreadsheet
        glider.createPseudogramSpreadsheet()
        #glider.stopToDebug()

        # Handle spreadsheet: send to stdout or write to file
        if args['csvOut']:
            glider.handleSpreadsheet()

        # Handle image: send to stdout or write to file
        if args['imageOut']:
            glider.handleImage()

        # Write netCDF file (if requested)
        if args['ncDir']:
            glider.writeNetCDF()

        # Write out the echogram bit information if requested
        if args['saveBits']:
            if 'echogram_bits' in glider.data.keys():
                fo = open(args['saveBits'], 'w')
                for gdata in glider.data['echogram_bits']:
                    tdata = gdata[0]
                    data = gdata[1:].astype('int')
                    fo.write('{:.3f} {:d} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}'.format(tdata, *data))
                    fo.write('\n')
                fo.close()
            else:
                print("WARNING: no echogram was found.")
