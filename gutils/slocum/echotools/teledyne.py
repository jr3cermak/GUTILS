import datetime
import dbdreader
import json
import io
import logging
import matplotlib
import os
import sys
import struct
import subprocess
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.dates as dates
import matplotlib.ticker as mticker
import xarray as xr
import pandas as pd

matplotlib.use('Agg')

# scale down logging for matplotlib and dbdreader
logging.getLogger("dbdreader").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)


class Glider:
    '''
    A container class for handling Teledyne Webb glider data.

    Table 7-1

    Glider : dbd mbd sbd mlg
    Science: ebd nbd tbd nlg
    '''

    def __init__(self, tbdFile = None, sbdFile = None, cacheDir = None, dbd2asc = None, debugFlag = False):
        '''
        Initialize a glider object.

        Parameters
        ----------
        tbdFile : :obj:`str`
            Full or relative path with filename to glider tbd file.
        sbdFile : :obj:`str`
            Full or relative path with filename to glider sbd file.
        cacheDir : :obj:`str`
            Full or relative path to directory with glider (sensor) cache files.
        dbd2asc : :obj:`str`
            Full or realtive path with filename to Teledyne Webb binary dbd2asc.
            NOTE: System must be able to execute the binary.
        debugFlag: :obj:`bool`
            Flag for extra debugging information printed to standard output.
            Default: False
        '''
        # Initialize object variables
        self.args = None
        self.tbdFile = tbdFile
        self.sbdFile = sbdFile
        self.cacheDir = cacheDir
        self.dbd2asc = dbd2asc
        self.debugFlag = debugFlag
        self.echotools = None
        self.ncDir = None
        self.ncUnlimitedDims = []
        self.fillValues = {}

        # IOOS/Local DAC metadata
        self.deployment = None
        self.instruments = None
        self.template = None
        self.dacOverlay = None

        # First attempt at structured data
        self.data = {
            'asc': None,
            'sbd': None,
            'echogram': None,
            'echogram_bits': None,
            'calibration': {},
            'columns': [],
            'byteSize': [],
            'units': [],
            'sbdcolumns': [],
            'sbdbyteSize': [],
            'sbdunits': [],
            'cacheMetadata': {
                'sensorCount': None,
                'factored': None,
                'stByteNum': None,
                'totalSensors': None,
                'cacheFile': None
            },
            'sbdMetadata': {
                'sensorCount': None,
                'factored': None,
                'stByteNum': None,
                'totalSensors': None,
                'cacheFile': None
            }
        }

        # Latest data structure
        self.data = {}
        self.data['cache'] = {}
        self.data['open'] = {}
        self.data['columns'] = {}
        self.data['units'] = {}
        self.data['input'] = {}
        self.data['process'] = None
        self.data['segment'] = None
        self.data['timestamp'] = {}
        self.data['inventory'] = None
        self.data['inventory_paths'] = []
        self.data['inventory_cache_path'] = None

        # Default mission parameters
        # Range units: meters
        self.mission_plan = {
            'bins': 20,
            'range': 60,
            'direction': -1,
            'bin_range': -3.0,
        }

        # Plotting
        self.availablePlotTypes = ['binned', 'scatter', 'pcolormesh', 'profile']
        self.defaultPlotType = 'binned'

        # Define colormaps
        self.cmaps = {}

        # Colormap 1: ek80
        # based on colors from https://github.com/EchoJulia/EchogramColorSchemes.jl
        ek80_colors = [
            [156 / 255, 138 / 255, 168 / 255],
            [141 / 255, 125 / 255, 150 / 255],
            [126 / 255, 113 / 255, 132 / 255],
            [112 / 255, 100 / 255, 114 / 255],
            [97 / 255, 88 / 255, 96 / 255],
            [82 / 255, 76 / 255, 78 / 255],
            [68 / 255, 76 / 255, 94 / 255],
            [53 / 255, 83 / 255, 129 / 255],
            [39 / 255, 90 / 255, 163 / 255],
            [24 / 255, 96 / 255, 197 / 255],
            [9 / 255, 103 / 255, 232 / 255],
            [9 / 255, 102 / 255, 249 / 255],
            [9 / 255, 84 / 255, 234 / 255],
            [15 / 255, 66 / 255, 219 / 255],
            [22 / 255, 48 / 255, 204 / 255],
            [29 / 255, 30 / 255, 189 / 255],
            [36 / 255, 12 / 255, 174 / 255],
            [37 / 255, 49 / 255, 165 / 255],
            [38 / 255, 86 / 255, 156 / 255],
            [39 / 255, 123 / 255, 147 / 255],
            [40 / 255, 160 / 255, 138 / 255],
            [41 / 255, 197 / 255, 129 / 255],
            [37 / 255, 200 / 255, 122 / 255],
            [30 / 255, 185 / 255, 116 / 255],
            [24 / 255, 171 / 255, 111 / 255],
            [17 / 255, 156 / 255, 105 / 255],
            [10 / 255, 141 / 255, 99 / 255],
            [21 / 255, 139 / 255, 92 / 255],
            [68 / 255, 162 / 255, 82 / 255],
            [114 / 255, 185 / 255, 72 / 255],
            [161 / 255, 208 / 255, 62 / 255],
            [208 / 255, 231 / 255, 52 / 255],
            [255 / 255, 255 / 255, 42 / 255],
            [254 / 255, 229 / 255, 43 / 255],
            [253 / 255, 204 / 255, 44 / 255],
            [253 / 255, 179 / 255, 45 / 255],
            [252 / 255, 153 / 255, 46 / 255],
            [252 / 255, 128 / 255, 47 / 255],
            [252 / 255, 116 / 255, 63 / 255],
            [252 / 255, 110 / 255, 85 / 255],
            [252 / 255, 105 / 255, 108 / 255],
            [252 / 255, 99 / 255, 130 / 255],
            [252 / 255, 93 / 255, 153 / 255],
            [252 / 255, 85 / 255, 160 / 255],
            [252 / 255, 73 / 255, 139 / 255],
            [253 / 255, 61 / 255, 118 / 255],
            [253 / 255, 48 / 255, 96 / 255],
            [254 / 255, 36 / 255, 75 / 255],
            [255 / 255, 24 / 255, 54 / 255],
            [240 / 255, 30 / 255, 52 / 255],
            [226 / 255, 37 / 255, 51 / 255],
            [212 / 255, 44 / 255, 50 / 255],
            [198 / 255, 51 / 255, 49 / 255],
            [184 / 255, 57 / 255, 48 / 255],
            [176 / 255, 57 / 255, 49 / 255],
            [170 / 255, 54 / 255, 51 / 255],
            [165 / 255, 51 / 255, 54 / 255],
            [159 / 255, 47 / 255, 56 / 255],
            [153 / 255, 44 / 255, 58 / 255],
            [150 / 255, 39 / 255, 56 / 255],
            [151 / 255, 31 / 255, 45 / 255],
            [153 / 255, 23 / 255, 33 / 255],
            [154 / 255, 15 / 255, 22 / 255],
            [155 / 255, 7 / 255, 11 / 255],
        ]
        ek80_cm = ListedColormap(ek80_colors)
        ek80_cm.set_over([255 / 255, 255 / 255, 255 / 255])
        ek80_cm.set_under([255 / 255, 255 / 255, 255 / 255])
        self.cmaps['ek80'] = ek80_cm

        # Colormap 2

        # Set the default SIMRAD EK500 color table plus grey for NoData.
        simrad_color_table = [
            (1, 1, 1),
            (0.6235, 0.6235, 0.6235),
            (0.3725, 0.3725, 0.3725),
            (0, 0, 1),
            (0, 0, 0.5),
            (0, 0.7490, 0),
            (0, 0.5, 0),
            (1, 1, 0),
            (1, 0.5, 0),
            (1, 0, 0.7490),
            (1, 0, 0),
            (0.6509, 0.3255, 0.2353),
            (0.4705, 0.2353, 0.1568)
        ]
        simrad_cmap = (
            LinearSegmentedColormap.from_list('simrad', simrad_color_table)
        )
        simrad_cmap.set_bad(color='lightgrey')
        #simrad_cmap.set_bad(color='k')
        self.cmaps['simrad'] = simrad_cmap

        self.defaultCmapType = 'ek80'
        self.availableCmapTypes = list(self.cmaps.keys())

    # General functions

    def appendFilenameSuffix(self, fname, word):
        '''
        Appends word to end of a filename.
        Ex: 20201213.csv => 20201213_{word}.csv
        '''

        fn, ext = os.path.splitext(fname)
        return f"{fn}_{word}{ext}"

    def calculateMissionPlan(self):
        '''
        Calculate mission plan parameters based on current arguments.
        '''

        # See if any parameters changed
        updateFlag = False
        if self.args.get('echogramBins'):
            if self.args['echogramBins'] != self.mission_plan['bins']:
                updateFlag = True
                self.mission_plan['bins'] = self.args['echogramBins']

        if self.args.get('echogramRange'):
            if self.args['echogramRange'] != self.mission_plan['range']:
                updateFlag = True
                if self.args['echogramRange'] >= 0:
                    self.mission_plan['direction'] = +1
                else:
                    self.mission_plan['direction'] = -1
                # Mission plan range is absolute value
                self.mission_plan['range'] = abs(self.args['echogramRange'])

        if updateFlag:
            self.mission_plan['bin_range'] = \
                (self.mission_plan['range'] / self.mission_plan['bins']) * \
                self.mission_plan['direction']
            if self.debugFlag:
                print("DEBUG: MISSION PLAN:", self.mission_plan)

    def createFileInventory(self, fileList, cache_dir):
        '''
        Create a slocum file inventory from a file list.  This list must include
        file extensions.
        '''

        # Create an empty pandas dataset
        columns = ['Start', 'End', 'File', 'Cache']
        df = pd.DataFrame(columns=columns)

        self.data['inventory'] = None
        self.data['inventory_paths'] = {}
        self.data['inventory_cache_path'] = cache_dir

        ct = -1
        for infile in fileList:
            abspath = os.path.abspath(os.path.dirname(os.path.relpath(infile)))
            if abspath not in self.data['inventory_paths']:
                ct = ct + 1
                plabel = f"PATH{ct:04d}"
                self.data['inventory_paths'][abspath] = plabel

            dbdFp = dbdreader.DBD(infile, cacheDir=cache_dir)
            dbdData = dbdFp.get(*dbdFp.parameterNames, return_nans=True)
            cacheFile = f"{dbdFp.cacheID}.cac"

            fileloc = self.data['inventory_paths'][abspath]
            fname = os.path.basename(infile)
            flabel = f"{fileloc}:{fname}"

            # Obtain time dimension and length of record
            data = dbdData[0][0]
            dlen = len(data)

            if dlen > 0:
                tmin = data.min()
                tmax = data.max()
            else:
                tmin = dbdFp.get_fileopen_time()
                tmax = None

            start_string = datetime.datetime.utcfromtimestamp(tmin).strftime("%Y-%m-%d %H:%M:%S")
            if tmax:
                end_string = datetime.datetime.utcfromtimestamp(tmax).strftime("%Y-%m-%d %H:%M:%S")
                if end_string == "nan":
                    end_string = "0000-00-00 00:00:00"
            else:
                end_string = "0000-00-00 00:00:00"

            rec = {
                'Start': start_string,
                'End': end_string,
                'File': flabel,
                'Cache': cacheFile
            }
            df = pd.concat([df, pd.Series(rec).to_frame().T], ignore_index=True)

        self.data['inventory'] = df
        return

    def dateFormat(self, dttmObj=datetime.datetime.utcnow(), fmt="%Y-%m-%dT%H:%M:%SZ"):
        '''
        Format a given datetime.datetime object into a string of the
        default format: "%Y-%m-%dT%H:%M:%SZ".  If a datetime.datetime object is
        not provided, use the current time.
        '''

        return datetime.datetime.strftime(dttmObj, fmt)

    def filterFiles(self, start_time=None, end_time=None):
        '''
        Filter a loaded list of slocum files by the start and end times.
        This requires use of a created or loaded file inventory.

        Parameters
        ----------
        start_time: :obj:`str`
            A string containing the start time.
        end_time: :obj:`str`
            A string containing the end time.

        Returns
        -------
        :obj:`list()`
            Returns a python list of matching files between the start and end times.
        '''

        if self.data['inventory'] is None:
            return []

        ds = self.data['inventory'].copy()
        # Convert 0000-00-00 00:00:00 to nan
        mask = ds['End'] == '0000-00-00 00:00:00'
        ds['End'][mask] = np.nan
        ds['Start_dt'] = ds['Start'].astype('datetime64', errors='ignore')
        ds['End_dt'] = ds['End'].astype('datetime64', errors='ignore')

        ds_start_time = pd.to_datetime(start_time).to_datetime64()
        ds_end_time = pd.to_datetime(end_time).to_datetime64()

        # Try to take a subset first
        mask1 = (ds['Start_dt'] >= ds_start_time) & (ds['End_dt'] <= ds_end_time) & (ds['End_dt'].notna())
        mask2 = (~ds['End_dt'].notna()) & (ds['Start_dt'] >= ds_start_time) & (ds['Start_dt'] <= ds_end_time)
        mask = (mask1 | mask2)
        df = ds.loc[mask]
        df_len = len(df)

        # If no results are found, attempt to find the nearest profile.
        # DEBUG
        #print(ds_start_time,ds_end_time,df_len)
        #breakpoint()
        if df_len == 0:
            idx_start = self.nearest(ds['Start_dt'], ds_start_time)
            df_start = ds.iloc[idx_start].to_frame().T

            idx_end = self.nearest(ds['End_dt'], ds_end_time)
            df_end = ds.iloc[idx_end].to_frame().T

            if idx_start != idx_end:
                # Select the closest of the two selected
                delta_start = np.abs(ds_start_time - (df_start['Start_dt'].to_numpy()))
                delta_end = np.abs(ds_end_time - (df_end['End_dt'].to_numpy()))
                if delta_start < delta_end:
                    df = df_start
                else:
                    df = df_end
            else:
                df = df_start

        # This does not work!
        #Convert nans back to 0000-00-00 00:00:00
        #mask = df['End'].isna()
        #df.loc[mask, 'End'] = '0000-00-00 00:00:00'
        #breakpoint()

        inv = Glider()

        # Pass 1
        inv.data['inventory'] = df
        inv.data['inventory_paths'] = self.data['inventory_paths']
        inv.data['inventory_cache_path'] = self.data['inventory_cache_path']
        fileList = inv.getFullFilenamesFromFileInventory()
        groupList = inv.groupFiles(fileList)

        # Pass 2
        # Resync subset inventory list with original inventory list
        # and pull through files that match the base file for
        # completeness.
        for gkey in groupList.keys():
            match = "%s\\." % (os.path.basename(gkey))
            ss1 = ds.loc[ds['File'].str.contains(match)]
            ss2 = df.loc[df['File'].str.contains(match)]
            df = pd.concat([df, ss1, ss2]).drop_duplicates()
        inv.data['inventory'] = df
        fileList = inv.getFullFilenamesFromFileInventory()
        groupList = inv.groupFiles(fileList)

        return groupList, inv

    def findTimeInterval(self, timeAxis, plotType, ax, nticks=10):
        '''
        For a given time axis and number of ticks, return
        an array of axisLocations and axisLabels.

        timeAxis is an array of timestamps in seconds.
        '''

        newLocs = []
        newLabels = []

        mint = timeAxis.min()
        maxt = timeAxis.max()
        tott = maxt - mint
        bint = tott / len(timeAxis)

        tickIntervals = {
            "1s": 1.0,
            "2s": 2.0,
            "5s": 5.0,
            "10s": 10.0,
            "20s": 20.0,
            "30s": 30.0,
            "1m": 60.0,
            "2m": 120.0,
            "5m": 300.0,
            "10m": 600.0,
            "20m": 1200.0,
            "30m": 1800.0,
            "1h": 3600.0,
            "2h": 7200.0,
            "3h": 10800.0,
            "6h": 21600.0,
            "12h": 43200.0,
            "1d": 86400.0,
            "2d": 172800.0,
            "3d": 259200.0,
            "5d": 432000.0,
            "10d": 864000.0,
            "20d": 1728000.0,
            "30d": 2592000.0,
        }

        # Determine interval that is nticks or less
        for intv in tickIntervals.keys():
            nsec = tickIntervals[intv]
            nint = tott / nsec
            if nint <= nticks:
                break

        if self.debugFlag:
            print("Selected time interval:", intv, nsec, nint)

        # Get the fractional portion of the selected interval
        # and go back one step to begin timestamping.
        fraction, whole = np.modf(timeAxis[0] / nsec)
        # If the fraction is small enough, just drop back an entire
        # interval
        if fraction < 0.005:
            fraction = (fraction * nsec) + nsec
        else:
            fraction = (fraction * nsec)
        taxis = timeAxis[0] - fraction

        # Test the time axis, if nsec >= 86400 and
        # we are at 00:00:00, don't show the time
        time_format_string = "%Y-%m-%d\n%H:%M:%S"
        if nsec >= 86400:
            ttime = datetime.datetime.utcfromtimestamp(taxis).strftime(time_format_string)
            if ttime[-8:] == "00:00:00":
                time_format_string = "%Y-%m-%d"

        if plotType != 'binned':
            # Convert time axis to matplotlib values
            #mTimeAxis = dates.date2num(pd.to_datetime(timeAxis,unit='s'))

            for i in range(0, int(nint) + 1):
                taxis = taxis + nsec
                ttime = datetime.datetime.utcfromtimestamp(taxis).strftime(time_format_string)
                tloc = dates.date2num(pd.to_datetime(taxis, unit='s'))

                newLabels.append(ttime)
                newLocs.append(tloc)

        else:
            # Otherwise, the time axis is in relation to pixel values
            # for imshow()

            # matplotlib is a bit mysterious when plotting binned data.
            # When given 56 xaxis points, the labels are arranged from
            # -10 to +60.

            for i in range(0, int(nint) + 1):
                taxis = taxis + nsec
                ttime = datetime.datetime.utcfromtimestamp(taxis).strftime(time_format_string)
                tloc = (taxis - timeAxis[0]) / bint

                newLabels.append(ttime)
                newLocs.append(tloc)

        return newLocs, newLabels

    def findTimeVariable(self, parameterList):
        '''
        Determine the time variable for a given parameter list.

        Parameters
        ----------
        parameterList: :obj:`list()`
            A python list of parameters.

        Returns
        -------
        :obj:`str`
            `m_present_time` or `sci_m_present_time` if detected.
            Returns None if either is not found.
        '''
        if 'm_present_time' in parameterList:
            return 'm_present_time'

        if 'sci_m_present_time' in parameterList:
            return 'sci_m_present_time'

        return None

    def getFullFilenamesFromFileInventory(self):
        '''
        Using the loaded file inventory, return full file paths.
        '''

        # Invert the inventory_paths dict()
        rpaths = {}
        for pkey in self.data['inventory_paths'].keys():
            rpaths[self.data['inventory_paths'][pkey]] = pkey

        full_file_list = []
        for index, row in self.data['inventory'].iterrows():
            fdata = row['File'].split(":")
            full_path = os.path.join(rpaths[fdata[0]], fdata[1])
            full_file_list.append(full_path)

        return full_file_list

    def groupFiles(self, fileList):
        '''
        Take a list of filenames and group them by name without thier extension.

        Parameters
        ----------
        array : :obj:`list()`
            A python list of filenames with extensions.  The case of the
            file extension is ignored.

        Returns
        -------
        :obj:`dict()`
            A python dictionary of file groups.
        '''

        groupList = {}
        for fname in fileList:
            fsplit = os.path.splitext(fname)
            fbase = fsplit[0]
            if fbase in groupList:
                groupList[fbase].append(fname)
            else:
                groupList[fbase] = [fname]

        return groupList

    def loadFileInventory(self, fname):
        '''
        This loads an existing file inventory of slocum files.

        Inventory file structure:
            Start, End, File, Cache
            Z_CACHE:....
            Z_PATH0000:....
        '''

        columns = ['Start', 'End', 'File', 'Cache']
        df = pd.DataFrame(columns=columns)

        self.data['inventory'] = df
        self.data['inventory_paths'] = {}
        self.data['inventory_cache_path'] = None

        fn = open(fname, 'r')
        for ln in fn:
            ln = ln.strip()
            if len(ln) > 0:
                if ln[0] == 'Z':
                    sdata = ln.split(":")
                    if sdata[0] == "Z_CACHE":
                        self.data['inventory_cache_path'] = sdata[1]
                    else:
                        pkey = sdata[0][2:]
                        self.data['inventory_paths'][sdata[1]] = pkey
                else:
                    sdata = ln.split(" ")
                    rec = {
                        'Start': "%s %s" % (sdata[0], sdata[1]),
                        'End': "%s %s" % (sdata[2], sdata[3]),
                        'File': sdata[4],
                        'Cache': sdata[5]
                    }
                    df = pd.concat([df, pd.Series(rec).to_frame().T], ignore_index=True)

        self.data['inventory'] = df

        return

    def loadMetadata(self):
        '''
        This function generically loads deployment and other metadata required
        for processing glider files into uniform netCDF files.
        '''
        if self.args.get('deploymentDir', None) is None:
            if self.args.get('ncDir', None):
                print("ERROR: A deployment configuration directory is required to write output to a netCDF file.")
                sys.exit()
            return
        else:
            if not os.path.isdir(self.args['deploymentDir']):
                print("ERROR: The deployment configuration directory was not found: %s" % (self.args['deploymentDir']))
                if self.args.get('ncDir', None) is None:
                    print("ERROR: A deployment configuration directory required to write output to a netCDF file.")
                sys.exit()

        # Attempt to read echotools.json configuration file
        try:
            echotoolsFile = os.path.join(self.args['deploymentDir'], 'echotools.json')
            testLoad = json.load(open(echotoolsFile))
            self.echotools = testLoad
        except Exception:
            print("WARNING: Unable to parse json echotools file: %s" % (echotoolsFile))
            #sys.exit()

        # Attempt to read deployment.json
        try:
            deploymentFile = os.path.join(self.args['deploymentDir'], 'deployment.json')
            testLoad = json.load(open(deploymentFile))
            self.deployment = testLoad
        except Exception:
            print("ERROR: Unable to parse json deployment file: %s" % (deploymentFile))
            sys.exit()

        # Attempt to read instruments.json
        try:
            instrumentsFile = os.path.join(self.args['deploymentDir'], 'instruments.json')
            testLoad = json.load(open(instrumentsFile))
            self.instruments = testLoad
        except Exception as err:
            print(f"ERROR: Unable to parse json instruments file: {instrumentsFile} {err=}")
            sys.exit()

        if self.args.get('ncDir', None):
            if self.args.get('templateDir', None) is None:
                print("ERROR: A template file is required to write output to a netCDF file.")
                sys.exit()

        if self.args.get('templateDir'):
            if not os.path.isdir(self.args['templateDir']):
                print("ERROR: The template directory was not found: %s" % (self.args['templateDir']))
                sys.exit()

            # Attempt to read <template>.json
            try:
                templateFile = os.path.join(self.args['templateDir'], self.args['template'])
                testLoad = json.load(open(templateFile))
                self.template = testLoad
            except Exception:
                print("ERROR: Unable to parse json template file: %s" % (templateFile))
                sys.exit()

        # Attempt to read auxillary metadata file (if specified)
        if self.args.get('dacOverlay', None):
            # Attempt to read <dacOverlay>.json
            try:
                dacOverlayFile = os.path.join(self.args['deploymentDir'], self.args['dacOverlay'])
                testLoad = json.load(open(dacOverlayFile))
                self.dacOverlay = testLoad
            except Exception:
                print("ERROR: Unable to parse json DAC overlay metadata file: %s" % (dacOverlayFile))
                sys.exit()

    def nearest(self, array, val):
        '''
        Find the nearest value in a sorted numpy array and return
        the index for the nearest value.

        Parameters
        ----------
        array : :obj:`numpy`
            A value sorted numpy array that is to be searched.
        val : :obj:`value`
            The value to search for the closest matching element in the provided numpy array.

        Returns
        -------
        :obj:`int`
            The index of the closest matching element.
        '''
        # This gets complicated when nan times are present
        array = np.asarray(array)
        idx = (np.abs(array - val))
        mask = np.isnan(idx)
        # Have nans?
        if np.any(mask):
            mask = ~np.isnan(idx)
            idx2 = idx[mask]
            tmin = idx2[idx2.argmin()]
            idx3 = np.argwhere(idx == tmin).flatten().tolist()
            # Use the first matching entry
            idx = idx3[0]
        else:
            idx = idx.argmin()

        return idx

    def saveFileInventory(self, invFile, sort_by_time=False):
        '''
        This saves a file inventory of slocum files.

        Inventory file structure:
            Start, End, File, Cache
            Z_CACHE:....
            Z_PATH0000:....
        '''

        if self.data['inventory'] is None:
            return

        if sort_by_time:
            self.data['inventory'] = self.data['inventory'].sort_values(by=['Start'])

        fo = open(invFile, 'w')
        for index, row in self.data['inventory'].iterrows():
            if pd.isna(row['End']):
                fo.write("%s %s %s %s\n" % (
                    row['Start'],
                    "0000-00-00 00:00:00",
                    row['File'],
                    row['Cache']
                ))
            else:
                fo.write("%s %s %s %s\n" % (
                    row['Start'],
                    row['End'],
                    row['File'],
                    row['Cache']
                ))

        if self.data['inventory_cache_path']:
            cache_dir = self.data['inventory_cache_path']
            fo.write(f"Z_CACHE:{cache_dir}\n")

        for ipath in self.data['inventory_paths'].keys():
            plabel = self.data['inventory_paths'][ipath]
            fo.write(f"Z_{plabel}:{ipath}\n")

        fo.close()

    def updateArgumentsFromMetadata(self):
        '''
        Adjust run time arguments based on metadata read by the glider class.
        This will only adjust arguments if self.deployment is set.
        '''

        if not self.deployment:
            if self.debugFlag:
                print("DEBUG: No deployment.conf file read.  No argument adjustments.")
            return self.args

        # From the deployment.json, we want to focus on the echograms entry.
        extra_kwargs = self.deployment.get('extra_kwargs')
        if extra_kwargs:
            echoconf = extra_kwargs.get('echograms')
            if echoconf:
                for ckey in echoconf.keys():
                    modKey = False
                    if ckey == 'plot_type':
                        if self.args.get('plotType', None) is None:
                            modKey = True
                            self.args['plotType'] = echoconf[ckey]
                    if ckey == 'plot_cmap':
                        if self.args.get('cmap', None) is None:
                            modKey = True
                            self.args['cmap'] = echoconf[ckey]
                    if ckey == 'svb_limits':
                        if self.args.get('dBLimits', None) is None:
                            modKey = True
                            self.args['dBLimits'] = echoconf[ckey]
                    if ckey == 'svb_thresholds':
                        if self.args.get('vbsBins', None) is None:
                            modKey = True
                            self.args['vbsBins'] = echoconf[ckey]
                    if ckey == 'echogram_range':
                        if self.args.get('echogramRange', None) is None:
                            modKey = True
                            self.args['echogramRange'] = echoconf[ckey]
                    if ckey == 'echogram_range_units':
                        if self.args.get('echogramRangeUnits', None) is None:
                            modKey = True
                            self.args['echogramRangeUnits'] = echoconf[ckey]
                    if ckey == 'echogram_range_bins':
                        if self.args.get('echogramBins', None) is None:
                            modKey = True
                            self.args['echogramBins'] = echoconf[ckey]

                    if self.debugFlag and modKey:
                        print("DEBUG: deployment.conf(%s) = %s" % (ckey, echoconf[ckey]))

        return self.args

    # Sorted function names here

    def stopToDebug(self):
        '''
        Generic function to stop python in its debugger.  When stopped by this
        function, it is necessary to go up one level in the execution stack to
        get to the exact location of the breakpoint.  Use the `up` command to
        go up one level in the execution stack.
        '''
        if self.debugFlag:
            try:
                breakpoint()
            except NameError:
                import pdb
                pdb.set_trace()

    def extractColumns(self, source, columns=[], ignoreNaNColumns=[], asDict=False):
        '''
        This function extracts requested columns from the GLIDER.data[source] object.
        This function will also remove rows for specified columns that contain NaNs.

        NOTE: This function replaces the need for dba_sensor_filter and subsequently
        ignoring output of NaNs.

        Parameters
        ----------
        columns : :obj:`list`
            Named columns to subset from GLIDER.data[source] object.
        ignoreNaNColumns : :obj:`list`
            One the data is collected by column, rows are eliminated in named
            columns where the values are NaN.
        asDict : :obj:`bool`
            This is a flag to change the return value as a python dict() object
            where the column names are the dictionary keys.

        Returns
        -------
        :obj:`numpy array or dict()`
            This returns a subset of data stored in GLIDER.data[source].  This will
            either be another numpy array or a python dictionary.
        '''

        colCount = 0
        dataColumns = []
        for col in columns:

            ind = -1
            try:
                if source == 'dbd':
                    ind = self.data['columns'][source].index(col)
                if source == 'ebd':
                    ind = self.data['columns'][source].index(col)
                if source == 'tbd':
                    ind = self.data['columns'][source].index(col)
                if source == 'sbd':
                    ind = self.data['columns'][source].index(col)
            except Exception:
                pass

            # Check for (1) non-existing column and no data
            if col not in self.data[source].keys():
                return None
            dataShape = self.data[source][col].shape
            if dataShape[0] == 0:
                return None

            if ind != -1:
                data = self.data[source][col]
                if colCount == 0:
                    selectedData = np.row_stack(data)
                    dataColumns.append(col)
                else:
                    selectedData = np.append(selectedData, np.row_stack(data), axis=1)
                    dataColumns.append(col)
                colCount = colCount + 1

        # Remove data rows with specified columns (ignoreNaNColumns)
        for col in ignoreNaNColumns:

            ind = -1
            try:
                ind = dataColumns.index(col)
            except Exception:
                pass

            if ind != -1:
                selectedData = selectedData[~np.isnan(selectedData[:, ind]), :]

        # Split columns into a dictionary if requested
        if asDict:
            colCount = 0
            dictData = {}
            for col in columns:
                dictData[col] = selectedData[:, colCount]
                colCount = colCount + 1
            selectedData = dictData

        return selectedData

    def handleSpreadsheet(self):
        '''
        This function handles writing out the decoded tbd data
        in CSV format.  Either a filename is provided or the
        spreadsheet is sent to 'stdout'.  If the csvHeader flag
        is set to True, a header is also provided.

        Notes for self.args

            * args['debugFlag']: Boolean flag.  If True, additional output is printed
              to standard output.
            * args['csvOut']: May be a full or relative path with filename or `stdout`.
            * args['csvHeader']: Boolean flag.  If True, a header is included with
              CSV output to file or standard out.
        '''

        # Skip if echogram is not present
        if self.data['spreadsheet'] is None:
            return

        # DEBUG
        if self.debugFlag:
            print("DEBUG: args.csvOut:", self.args['csvOut'])
            print("DEBUG: args.csvHeader:", self.args['csvHeader'])
            print("DEBUG: args.outDir:", self.args['outDir'])

        # If this argument is None, do not run this function
        if self.args['csvOut'] is None:
            return

        # If args.csvOut is not "stdout" assume this is a filename
        outptr = sys.stdout
        stdoutFlag = True

        # Skip output if file fails to open for writing
        if self.args['csvOut'] != "stdout":

            # See if we need to substitute the filename if the default
            # is provided.
            outFilename = self.args['csvOut']
            if 'default' in outFilename:
                timeToUse = None
                if 'tbd' in self.data['timestamp']:
                    timeToUse = self.data['timestamp']['tbd']
                if 'ebd' in self.data['timestamp']:
                    timeToUse = self.data['timestamp']['ebd']

                outFilename = outFilename.replace(
                    'timestamp',
                    self.dateFormat(
                        datetime.datetime.utcfromtimestamp(timeToUse), fmt="%Y%m%d_%H%M%S"
                    )
                )

            # One last adjustment to CSV if reprocess is being used
            if self.data['process']:
                outFilename = self.appendFilenameSuffix(outFilename, self.data['process'])

            # Use the input filename as the csv output filename
            try:
                if self.args['outDir']:
                    outptr = open(os.path.join(self.args['outDir'], outFilename), "w")
                else:
                    outptr = open(outFilename, "w")
                stdoutFlag = False
            except Exception:
                print("WARNING: Unable to open CSV output file (%s) for writing, skipping." % (self.args['csvOut']))
                return

        if self.args['csvHeader']:
            outptr.write("Timestamp(seconds since 1970-1-1), Depth(meters), Sv(dB)\n")

        #self.stopToDebug()
        for data in self.data['spreadsheet']:
            outptr.write("%f, %f, %f\n" % (data[0], data[1], data[2]))

        if not stdoutFlag:
            outptr.close()

        return

    def getDepthPixel(self, reqDepth, minDepth, maxDepth, depthBinSize):
        '''
        For the image (plotting) routine, depths are placed
        in discrete pixels by the depth bin size.  The first
        pixel has the depth range of minimum depth to minimum
        depth plus depth bin size.

        Notes
        -----
            For a depth bin size of 3.0 meters and a minimum depth of
            2.0 meters.  Bin zero (0) should be 2.0 to 5.0 meters, bin
            (1) will be from 5.0 meters to 8.0 meters and so forth.

        Parameters
        ----------
        reqDepth : :obj:`float`
            Requested depth (meters)
        minDepth : :obj:`int`
            Minimum depth bin
        maxDepth : :obj:`int`
            Maximum depth bin (not used)
        depthBinSize : :obj:`float`
            Actual depth size of each depth bin (meters)

        Returns
        -------
        :obj:`int`
            Returns the depth bin adjusted to the minimum depth bin

        '''
        idx = int(reqDepth / depthBinSize) - minDepth
        return idx

    def handleImage(self):
        '''
        This function handles writing out a graphical image.  By
        default, the image rendering uses descrete pixels that
        have depth and time bins.  If --useScatterPlot is set,
        a scatter plot is produced instead of a time/depth binned
        (raster image/imshow) plot.  The raster plot coordinates
        are the depth and time bins which requires redefining x
        and y labels on the fly.

        Notes for self.args

            * args['imageOut']: May be a full or relative path with filename or `stdout`.
            * args['debugFlag']: Boolean flag.  If True, additional output is printed
              to standard output.
            * args['useScatterPlot']: Boolean flag.  If True, a scatter plot is
              produced instead of the depth/time binned plot.
        '''

        # Skip plot if data is missing
        if self.data['spreadsheet'] is None:
            return

        # If args.imageOut is not "stdout" assume this is a filename
        stdoutFlag = True

        if self.args.get('imageOut', None):
            if self.args['imageOut'] != "stdout":
                stdoutFlag = False

        #depthBinSize = int(abs(self.data['depthBinLength']))
        depthBinSize = int(abs(self.mission_plan['bin_range']))
        timeBinSize = self.data['timeBinLength']

        if self.debugFlag:
            print("PLOTTING: timeBinSize=", timeBinSize)

        # Spreadsheet columns
        # [0] Timestamp(seconds since 1970-1-1, [1] depth(meters), [2] Sv (dB)
        dataSpreadsheet = self.data['spreadsheet']

        if dataSpreadsheet is None:
            print("WARNING: No usable data found!")
            if self.sbdFile is None:
                print("HINT: Include a sbd file if available.")
            return

        # We need to know the time indexes for the graphics below
        timeIndexes = np.unique(dataSpreadsheet[:, 0])

        # Determine plot type requested
        # Default: binned
        plotTypes = [self.defaultPlotType]
        if 'plotType' in self.args:
            plotTypes = []
            reqPlots = self.args['plotType'].split(',')
            if self.debugFlag:
                print("DEBUG: reqPlots:", reqPlots)
            if 'all' in reqPlots:
                reqPlots = self.availablePlotTypes
            for reqPlot in reqPlots:
                if reqPlot in self.availablePlotTypes:
                    if reqPlot not in plotTypes:
                        plotTypes.append(reqPlot)
                # If stdout is selected, only process the first requested
                # plot
                if stdoutFlag and len(plotTypes) == 1:
                    break

        if self.debugFlag:
            print("DEBUG: Selected plot type(s):", plotTypes)

        # Determine color map requested
        cmap = self.cmaps[self.defaultCmapType]
        if 'cmap' in self.args:
            if self.args['cmap'] in self.availableCmapTypes:
                cmap = self.cmaps[self.args['cmap']]

        # Default colorbar ylabel and size
        #default_cb_ylabel = r'Sv (dB re 1 $\bf{m^2}$/$\bf{m^3}$)'
        default_cb_ylabel = r'$\bf{Sv}$ $\bf{(dB}$ $\bf{re}$ $\bf{1}$ $\bf{m^2}$/$\bf{m^3}$$\bf{)}$'
        default_cb_shrink = 0.60

        # Default plot parameters

        # Calculate how many depth pixels (bins) we need given spreadsheet data
        # minDepth = shallowest pixel
        # maxDepth = deepest pixel
        minDepth = int(dataSpreadsheet[:, 1].min() / depthBinSize)
        maxDepth = int(np.ceil(dataSpreadsheet[:, 1].max() / depthBinSize))

        if self.debugFlag:
            print("Depth bins: Min(%d) Max(%d)" % (minDepth, maxDepth))

        numberOfDepthPixels = (maxDepth - minDepth) + 1
        depthTicks = []
        for depthPixel in range(0, numberOfDepthPixels, 5):
            startDepth = minDepth * depthBinSize + depthPixel * depthBinSize
            endDepth = startDepth + depthBinSize
            midPixelDepth = (startDepth + endDepth) / 2.0
            depthTicks.append(midPixelDepth)

        # Setup rigid colorbar and plotting limits
        dB_limit = (self.data['echogram_bins'][0], self.data['echogram_bins'][7])
        if 'dBLimits' in self.args:
            dB_limit = self.args['dBLimits']

        norm = mpl.colors.Normalize(vmin=dB_limit[1], vmax=dB_limit[0])
        #breakpoint()

        for plotType in plotTypes:

            # Determine output filename

            plot_title = None
            if self.args['title']:
                plot_title = self.args['title']

            # See if we need to substitute strings in the filename or
            # plot title.

            # {plottype} => ['binned', 'scatter', 'profile', 'pcolormesh']
            # {segment} => unit_507-2023-085-2-99
            # {timestamp} =>
            #   Filename: yyyymmdd_hhmmss
            #   Title: yyyy/mm/dd hh:mm:ss
            outFilename = self.args['imageOut']
            timeToUse = None
            if 'tbd' in self.data['timestamp']:
                timeToUse = self.data['timestamp']['tbd']
            if 'ebd' in self.data['timestamp']:
                timeToUse = self.data['timestamp']['ebd']

            if outFilename:
                if 'plottype' in outFilename:
                    outFilename = outFilename.replace('plottype', plotType)
                if 'segment' in outFilename:
                    outFilename = outFilename.replace('segment', self.data['segment'])
                if 'timestamp' in outFilename:
                    outFilename = outFilename.replace(
                        'timestamp',
                        self.dateFormat(
                            datetime.datetime.utcfromtimestamp(timeToUse),
                            fmt="%Y%m%d_%H%M%S")
                    )

            if plot_title:
                if 'plottype' in plot_title:
                    plot_title = plot_title.replace('plottype', plotType)
                if 'segment' in plot_title:
                    plot_title = plot_title.replace('segment', self.data['segment'])
                if 'timestamp' in plot_title:
                    plot_title = plot_title.replace(
                        'timestamp',
                        self.dateFormat(
                            datetime.datetime.utcfromtimestamp(timeToUse),
                            fmt="%Y/%m/%d %H:%M:%S"))

            # One last adjustment to filename if reprocess is being used
            if self.data['process']:
                outFilename = self.appendFilenameSuffix(outFilename, self.data['process'])

            """
            if 'default' in outFilename:
                if self.data['segment']:
                    outFilename = outFilename.replace('default', "%s_%s" % \
                        (self.data['segment'], plotType))
                else:
                    timeToUse = None
                    if 'tbd' in self.data['timestamp']:
                        timeToUse = self.data['timestamp']['tbd']
                    if 'ebd' in self.data['timestamp']:
                        timeToUse = self.data['timestamp']['ebd']
                    if timeToUse:
                        outFilename = outFilename.replace('default',
                            "%s_%s" % (self.dateFormat(datetime.datetime.utcfromtimestamp(timeToUse),
                                fmt="%Y%m%d_%H%M%S"), plotType))
                    else:
                        outFilename = outFilename.replace('default', plotType)

                if plot_title:
                    if 'default' in plot_title:
                        plot_title = plot_title.replace('default', plotType)
            """

            # pcolormesh plot
            if plotType in ['pcolormesh']:

                unique_times = np.unique(dataSpreadsheet[:, 0])
                # matrix size = (time, bins)
                # matrix       => (matrix units)
                # (xx, yy, zz) => (time, depth, dB)
                xx = np.repeat(unique_times, self.mission_plan['bins'])
                yy = np.zeros((self.mission_plan['bins'], len(unique_times)))
                zz = yy.copy()

                tindex = -1
                for tm in unique_times:
                    tindex = tindex + 1
                    mask = (dataSpreadsheet[:, 0] == tm)
                    yyy = dataSpreadsheet[mask, 1]
                    yy[0:len(yyy), tindex] = yyy
                    # If not all the bins are represented, we have to add
                    # the depth shift to the missing bins just to make the
                    # matrix complete.
                    for i in range(len(yyy), self.mission_plan['bins']):
                        yy[i, tindex] = yy[i - 1, tindex] + self.mission_plan['bin_range']
                    zzz = dataSpreadsheet[mask, 2]
                    zz[0:len(zzz), tindex] = zzz
                    # If not all the bins are represented, fill blanks with NaN
                    for i in range(len(zzz), self.mission_plan['bins']):
                        zz[i, tindex] = np.nan

                xx = np.array(pd.to_datetime(xx.ravel(), unit='s'))\
                    .reshape(len(unique_times), self.mission_plan['bins']).T

                fig, ax = plt.subplots(figsize=(5, 4))

                #ax.xaxis.set_minor_locator(dates.MinuteLocator(interval=10))# every 10 minutes
                #ax.xaxis.set_minor_formatter(dates.DateFormatter('%H:%M'))  # hours and minutes
                #ax.xaxis.set_major_locator(dates.DayLocator(interval=1))    # every day
                #ax.xaxis.set_major_formatter(dates.DateFormatter('\n%m-%d-%Y'))

                if 'dBLimits' in self.args:
                    ax.pcolormesh(
                        xx, yy, zz, shading='nearest',
                        vmin=dB_limit[1], vmax=dB_limit[0],
                        cmap=cmap,
                    )
                else:
                    ax.pcolormesh(
                        xx, yy, zz, shading='nearest',
                        cmap=cmap,
                    )

                ax.set_facecolor('lightgray')
                ax.set_ylim(0)
                ax.set_ylabel('Depth (m)')
                ax.set_xlabel('Date/Time (UTC)')
                ax.invert_yaxis()

                # Adjust x ticks
                xtickLocs, xtickLabels = self.findTimeInterval(timeIndexes, plotType, ax, nticks=5)
                ax.xaxis.set_major_locator(mticker.FixedLocator(xtickLocs))
                ax.set_xticklabels(xtickLabels)
                plt.xticks(rotation = 45.0)

                # Color bar using vmin,vmax
                #cx = fig.colorbar(px, ticks=dB_ticks)
                #cx = fig.colorbar(px, shrink=default_cb_shrink)
                cx = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), shrink=default_cb_shrink)
                cx.ax.get_yaxis().labelpad = 15
                cx.ax.set_ylabel(default_cb_ylabel)

                if self.debugFlag:
                    print("DEBUG: End of pcolormesh plotting routine.")

            # profile plots are just depth vs bin!
            if plotType == 'profile':

                dot_size = 70.0

                scatterData = dataSpreadsheet.copy()

                # Replace time index with bin #
                for timeIdx in timeIndexes:
                    cond = np.where(scatterData[:, 0] == timeIdx)[0]
                    cond_min = cond.min()
                    scatterData[cond, 0] = cond - cond_min

                # Plot higher reflective echoes on top of plot
                scatterData = scatterData[np.argsort(scatterData[:, 2])]

                # Mask any high data points that plot white over the image
                # data.  Any pts < dB_limit[0]
                mask = scatterData[:, 2] > dB_limit[0]
                scatterData[mask, 2] = np.nan

                fig, ax = plt.subplots(figsize=(3, 6))

                plt.scatter(
                    scatterData[:, 0], scatterData[:, 1], c=scatterData[:, 2],
                    cmap=cmap, norm=norm, s=dot_size)

                plt.colorbar(
                    mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                    ax=plt.gca(),
                    orientation='vertical', label=default_cb_ylabel, shrink=default_cb_shrink)
                plt.gca().invert_yaxis()
                plt.ylabel('depth (m)')

                gca_ax = plt.gca()
                gca_ax.get_xaxis().set_visible(False)

                if self.debugFlag:
                    print("DEBUG: End of profile plot routine.")

            # scatter plot
            if plotType in ['scatter']:

                dot_size = 40.0

                # Sort Sv(dB) from lowest to highest so higher values are plotted last
                scatterData = dataSpreadsheet[np.argsort(dataSpreadsheet[:, 2])]

                # Plot simply x, y, z data (time, depth, dB)
                fig, ax = plt.subplots(figsize=(5, 4))
                #ax.xaxis.set_minor_locator(dates.MinuteLocator(interval=10))# every 10 minutes
                #ax.xaxis.set_minor_formatter(dates.DateFormatter('%H:%M'))  # hours and minutes
                #ax.xaxis.set_major_locator(dates.DayLocator(interval=1))    # every day
                #ax.xaxis.set_major_formatter(dates.DateFormatter('\n%m-%d-%Y'))
                ax.set_facecolor('lightgray')
                plt.scatter(
                    pd.to_datetime(scatterData[:, 0], unit='s'), scatterData[:, 1], c=scatterData[:, 2],
                    cmap=cmap, norm=norm, s=dot_size)
                #cbar = plt.colorbar(orientation='vertical', label=default_cb_ylabel, shrink=default_cb_shrink)
                plt.colorbar(
                    mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                    orientation='vertical', label=default_cb_ylabel, shrink=default_cb_shrink)
                plt.gca().invert_yaxis()
                plt.ylabel('depth (m)')
                plt.xlabel('Date/Time (UTC)')

                # Adjust x ticks
                xtickLocs, xtickLabels = self.findTimeInterval(timeIndexes, plotType, ax, nticks=5)
                ax.xaxis.set_major_locator(mticker.FixedLocator(xtickLocs))
                ax.set_xticklabels(xtickLabels)
                plt.xticks(rotation = 45.0)

                if self.debugFlag:
                    print("DEBUG: End of scatter plot routine.")

            # binned plot
            if plotType in ['binned']:

                # Loop through each time on the spreadsheet
                # Each scan is a pre-generated array with -90.0 dB values (above most thresholds)
                numDataRecords = 0
                for timeIdx in timeIndexes:
                    sample = np.ones((numberOfDepthPixels,)) * -90.0

                    cond = np.where(dataSpreadsheet[:, 0] == timeIdx)[0]
                    for recIdx in cond:
                        (timeRec, depthRec, dBRec) = dataSpreadsheet[recIdx, :]
                        depthBin = self.getDepthPixel(depthRec, minDepth, maxDepth, depthBinSize)

                        # DEBUG
                        #if self.debugFlag:
                        #    print("  Timestamp(%.1f) Depth(%.1f meters) Sv(%.1f dB) DepthPixel(%d)" %\
                        #            (timeRec, depthRec, dBRec, depthBin))

                        sample[depthBin] = dBRec
                    if numDataRecords == 0:
                        imageData = sample
                    else:
                        imageData = np.vstack((imageData, sample))
                    numDataRecords = numDataRecords + 1

                # Create image plot
                # First we change the pixels with the default -90.0 dB
                # value to NANs so they become transparent.
                imageData[np.where(imageData == -90.0)] = np.nan

                # Axis labels are centered on the pixel
                fig, ax = plt.subplots(figsize=(5, 4))

                plotData = np.transpose(imageData)
                plotDataShape = plotData.shape
                if len(plotDataShape) == 1:
                    if self.debugFlag:
                        print("WARNING: Not enough pings to produce binned plot.")
                    return
                #breakpoint()
                plt.imshow(plotData, cmap=cmap, interpolation='none')
                plt.clim(dB_limit[1], dB_limit[0])
                plt.colorbar(
                    mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                    ax=plt.gca(),
                    orientation='vertical', label=default_cb_ylabel, shrink=default_cb_shrink)

                # x and y axis labels
                plt.ylabel('depth (m)')
                plt.xlabel('Date/Time (UTC)')
                #plt.clim(0, -60)
                #plt.clim(0, self.data['echogram_bins'][7])
                #print(self.data['echogram_bins'])
                #plt.clim(dB_limit[0], dB_limit[1])

                # Adjust x ticks
                xtickLocs, xtickLabels = self.findTimeInterval(timeIndexes, plotType, ax, nticks=5)
                ax.xaxis.set_major_locator(mticker.FixedLocator(xtickLocs))
                ax.set_xticklabels(xtickLabels)
                plt.xticks(rotation = 45.0)

                # Adjust y tick labels: depth bin -> depth (meters)
                ytickLabels = []
                ytickLocs = ax.get_yticks()
                for depthTick in ytickLocs:
                    startDepth = minDepth * depthBinSize + (depthTick * depthBinSize)
                    endDepth = startDepth + depthBinSize
                    midPixelDepth = (startDepth + endDepth) / 2.0
                    ytickNew = ("%.1f" % (midPixelDepth))
                    ytickLabels.append(ytickNew)

                if not self.args.get('binnedDepthLabels', False):
                    # Ensure y-axis is limited to about 10 major ticks.
                    depthIntervals = [10.0, 20.0, 25.0, 50.0, 100.0, 150.0, 200.0]

                    # We really do not need the first and last values
                    # of the ytickLocs.  We also need to find the slope
                    # to calculate the actual depth.
                    ytickLocs = ytickLocs[1:-1]
                    ytickLabels = ytickLabels[1:-1]
                    floatLabels = np.array(ytickLabels, float)
                    # Slope is based on the range difference represented in
                    # the two refrences.
                    binToDepthSlope = (ytickLocs[-1] - ytickLocs[0]) / \
                        (floatLabels[-1] - floatLabels[0])

                    # Determine depth interval to use
                    # Use an interval that produces about 10 ticks
                    fullDepth = (maxDepth - minDepth) * depthBinSize
                    selectedDepthInterval = 10.0
                    for interval in depthIntervals:
                        nIntervals = int(fullDepth / interval)
                        selectedDepthInterval = interval
                        if nIntervals <= 10:
                            break

                    #self.stopToDebug()
                    ytickLabels = []
                    ytickLocs = []
                    yOffset = floatLabels[0] * binToDepthSlope
                    depth = - selectedDepthInterval
                    while depth < (maxDepth * depthBinSize):
                        depth = depth + selectedDepthInterval
                        ytickLoc = depth * binToDepthSlope - yOffset
                        ytickLocs.append(ytickLoc)
                        ytickLabels.append("%d" % (depth))

                    #self.stopToDebug()

                ax.yaxis.set_major_locator(mticker.FixedLocator(ytickLocs))
                ax.set_yticklabels(ytickLabels)

                if self.debugFlag:
                    print("DEBUG: End of binned plotting routine.")

            # Plot title is generic
            if plot_title:
                plt.title(plot_title)

            # Determine if we are writing to stdout
            if stdoutFlag:
                plt.savefig(sys.stdout.buffer, dpi=100)
            else:
                # Plot image
                if self.args['outDir']:
                    if self.debugFlag:
                        print("DEBUG: Wrote to", os.path.join(self.args['outDir'], outFilename))
                    plt.savefig(os.path.join(self.args['outDir'], outFilename), dpi=100, bbox_inches='tight')
                    plt.close()
                else:
                    if self.debugFlag:
                        print("DEBUG: Wrote to", outFilename)
                    plt.savefig(outFilename, dpi=100, bbox_inches='tight')
                    plt.close()

    # Glider functions

    def createEchogramSpreadsheet(self):
        '''
        This function reads GLIDER.data['echogram'] and
        places it in a spreadsheet format in
        GLIDER.data['spreadsheet'].  The function is
        expecting at least two fields to have been read
        from the provided ebd/tbd file.

        Notes for self.args

            * args['debugFlag']: Boolean flag.  If True, additional
              output is printed to standard output.
            * args['plotType']: String.  Available plot types are:
              binned, scatter, pcolormesh or profile.
        '''

        # Code from echoGenNew.py
        eData = self.data['echogram']
        #print(self.data['echogram_bits'][0][1:])
        #breakpoint()

        # Skip if there isn't a echogram
        if eData is None:
            self.data['spreadsheet'] = None
            return

        # We extract the columns from decoded dbd2asc instead of
        # requiring another Teledyne program (dba_sensor_filter)
        if self.debugFlag:
            print("First depth data source:", self.data['echogram_source'])
        barData = self.extractColumns(
            self.data['echogram_source'], columns=['sci_m_present_time', 'sci_water_pressure'],
            ignoreNaNColumns=['sci_water_pressure'])

        # Bar data may not be available from the tbd file, ether there will be zero rows
        # or lack of columns.
        if barData is not None:
            dataShape = barData.shape
            if dataShape[1] < 2 or dataShape[0] == 0:
                barData = None

        # Change bars to meters
        if barData is not None:
            barData[:, 1] = barData[:, 1] * 10.0

        # If dbd data is available, merge the dbd data with the
        # ebd barData.
        # If sbd data is available, merge the sbd data with the
        # tbd barData.
        source_depth_data = 'dbd'
        depthData = None
        if source_depth_data not in self.data.keys():
            if 'sbd' in self.data.keys():
                source_depth_data = 'sbd'
        if self.debugFlag:
            print("Source depth data:", source_depth_data)
        if source_depth_data in self.data.keys():
            depthData = self.extractColumns(
                source_depth_data, columns=['m_present_time', 'm_depth'],
                ignoreNaNColumns=['m_depth'])
            if barData is not None:
                #self.stopToDebug()
                barData = np.append(barData, depthData, axis=0)
            else:
                barData = depthData
            #self.stopToDebug()
            if barData is not None:
                barData = barData[np.argsort(barData[:, 0])]

        if barData is None:
            print("WARNING: No usable depth information found for %s" % (self.args['sbdFile']))
            self.data['depthBinLength'] = None
            self.data['timeBinLength'] = None
            self.data['spreadsheet'] = None
            return

        # Separate time from ping (echosound) data
        pingTime = eData[:, 0]
        # Columns 1 -> 21
        #   Assuming values are increasing with depth
        pingData = eData[:, 1:]

        # Separate time from depth (bar) data
        depthTimes = barData[:, 0]
        # Separate depth from depth (bar) data
        depths = barData[:, 1]

        # Echogram range (meters)
        # Positive values: instrument is pointed down
        # Negative values: instrument is pointed up
        echogramRange = float(self.args['echogramRange'])
        # Number of depth bins (0-19)
        numberDepthBins = 20
        #depthBins = range(0, numberDepthBins)
        # Currently fixed at 3.0 meters
        depthBinLength = echogramRange / numberDepthBins

        # Determine time range to plot based on depthTimes
        if len(depthTimes) == 0:
            print("WARNING: No usable depth information found!")
            self.data['depthBinLength'] = None
            self.data['timeBinLength'] = None
            self.data['spreadsheet'] = None
            return

        # Average time between depths (should be about 8 seconds)
        # If we assume 8.0 seconds, then the time bin is
        # timeBin - 4.0 seconds to timeBin + 4.0 seconds

        # This is the size of the time bin in which pings can be
        # collected.
        avgDepthReading = ((depthTimes[-1] - depthTimes[0]) / len(depthTimes)) / 2.0
        timeBinLength = avgDepthReading * 2.0

        # Our time bins to collect pings extend by one average
        minDepthTime = depthTimes[0] - avgDepthReading
        maxDepthTime = depthTimes[-1] + avgDepthReading
        numberTimeSamples = int((maxDepthTime - minDepthTime) / (avgDepthReading * 2.0))
        if self.debugFlag:
            print("Time dimension:")
            print("  Start time(min): %s (%f)" % (datetime.datetime.utcfromtimestamp(minDepthTime), minDepthTime))
            print("  End   time(max): %s (%f)" % (datetime.datetime.utcfromtimestamp(maxDepthTime), maxDepthTime))
            print("  Avg time interval: %f seconds" % (avgDepthReading * 2.0))
            print("  Max offset +/- seconds: %f" % (avgDepthReading))
            print("  Number of time samples: %d (%d)" % (numberTimeSamples, len(depthTimes)))

        # Pings that fall within the time bins
        usablePings = np.where(np.logical_and(pingTime >= minDepthTime, pingTime <= maxDepthTime))
        numberUsablePings = len(usablePings[0])
        if self.debugFlag:
            print("Number of usable pings:", numberUsablePings)

        # Determine depth range to plot
        minDepth = depths.min()
        maxDepth = depths.max()
        if self.debugFlag:
            print("Depth dimension:")
            print("  Depth: min(%f) max(%f)" % (minDepth, maxDepth))
            print("  Instrument range: %f meters" % (echogramRange))
            print("  Depth bin size: %f meters" % (depthBinLength))

        # Run through each of the scans and prepare a data table/spreadsheet

        # If a time index on a set of pings is the same, increment the
        # plotting time index by 10 seconds to obtain proper alignment.
        #self.stopToDebug()
        if self.debugFlag:
            print("Processing pings:")

        # Time (seconds), Depth (meters), Sv (dB)
        dataSpreadsheet = None
        dataCount = 0
        lastPingTime = -1
        pingOffset = 0.0

        for pingNo in usablePings[0]:

            # Find the time index for this ping in the depth data
            depthTimeIndex = self.nearest(depthTimes, pingTime[pingNo])
            depthTime = depthTimes[depthTimeIndex]
            selectedDepth = depths[depthTimeIndex]
            timeOffset = pingTime[pingNo] - depthTime

            # Skip pings that do not have a matching depth reading within the time
            # tolerance of the available depth data
            if abs(timeOffset) > (avgDepthReading * 2.0):
                if self.debugFlag:
                    print(
                        "  WARNING: Ping (%d) skipped: Ping time (%f) Offset (%f seconds) > Time Bin (%f seconds)" %
                        (pingNo, pingTime[pingNo], timeOffset, avgDepthReading))
                continue

            # If this ping is the same time, then we need to apply a 10 second offset to
            # the new set of ping data.  This is for the "echo" mode.
            if lastPingTime == pingTime[pingNo]:
                pingOffset = pingOffset + 10.0
            else:
                pingOffset = 0.0
                lastPingTime = pingTime[pingNo]

            #if self.debugFlag:
            #    print("  Ping (%d) Ping time (%f) Depth time (%f) Offset (%f seconds) Depth (%f meters)" % \
            #        (pingNo, pingTime[pingNo]+pingOffset, depthTime, timeOffset, selectedDepth))

            # Process ping data is always closest to farthest (0, maxRange)
            # maxRange (-) if instrument is pointed upwards (shallower)
            # maxRange (+) if instrument is pointed downwards (deeper)
            pingDepth = selectedDepth - depthBinLength
            for ping in pingData[pingNo]:
                pingDepth = pingDepth + depthBinLength

                # If pingDepth is above the surface, ignore it
                if pingDepth < 0.0:
                    continue

                #if self.debugFlag:
                #    print("      Sv (%f dB) depth (%f meters)" % (ping, pingDepth))

                dataRow = np.array((pingTime[pingNo] + pingOffset, pingDepth, ping))
                if dataCount == 0:
                    dataSpreadsheet = np.column_stack(dataRow)
                else:
                    dataSpreadsheet = np.append(dataSpreadsheet, np.column_stack(dataRow), axis=0)
                dataCount = dataCount + 1

        self.data['depthBinLength'] = depthBinLength
        self.data['timeBinLength'] = timeBinLength
        self.data['spreadsheet'] = dataSpreadsheet

        return

    def readDbd(self, **kwargs):
        '''
        This function reads any DBD glider file using the dbdreader python library.

        :param \\**kwargs:
            See below
        :return: xarray Dataset or python dictionary
        :rtype: xarray Dataset() or dict()

        **Keyword Arguments**

            * *inputFile* (``string``) -- relative or full path to glifer DBD file
            * *cacheDir* (``string``) -- relative or full path to glider cache
              directory.  This overrides the default class object value.
            * *returnDict* (``boolean``) -- when set True, the data returned from
              this function is a python dictionary. Default: **False**
        '''

        # Process keyword arguments
        inputFile = None
        if 'inputFile' in kwargs.keys():
            inputFile = kwargs['inputFile']
        cacheDir = self.cacheDir
        if 'cacheDir' in kwargs.keys():
            cacheDir = kwargs['cacheDir']
        returnDict = False
        if 'returnDict' in kwargs.keys():
            returnDict = kwargs['returnDict']

        if not os.path.isfile(inputFile):
            print("ERROR: Glider DBD file not found: %s" % (inputFile))
            sys.exit()

        (fpath, fext) = os.path.splitext(inputFile)
        self.data['segment'] = os.path.basename(fpath)
        dbdType = fext[1:].lower()

        # Skip these file types for now
        if dbdType in ['mbd', 'nbd', 'mlg', 'nlg']:
            return

        if self.debugFlag:
            print("Reading file type:", dbdType)

        if not returnDict:
            # Return a xarray Dataset() object
            dataObj = xr.Dataset()
        else:
            # Return a python dictionary object
            dataObj = dict()

        #dbdFp = dbdreader.DBD(inputFile, cacheDir=cacheDir)
        dbdFp = None
        try:
            dbdFp = dbdreader.DBD(inputFile, cacheDir=cacheDir)
        except Exception:
            print("WARNING: Unable to read glider DBD file: %s" % (inputFile))

        if dbdFp:
            # Perform the actual read here
            # Collect all parameters. To retain the shape of the dataset, ask for NaNs.
            dbdData = dbdFp.get(*dbdFp.parameterNames, return_nans=True)

            # This returns data that is suppose to be prefixed with time but it will
            # be corrupted. The format of the data is:
            #     dbdData[column][0=time; 1=data value][row]

            # Obtain time dimension
            timeDimension = self.findTimeVariable(dbdFp.parameterNames)
            if timeDimension is None:
                sys.exit("ERROR: no time variable found in DBD file (%s)" % (inputFile))

            #NBD
            #python -m pdb -c continue ./scanDBDFiles.py --cacheDir sfmc/unit_507/cache --input raw/unit_507/20220212/sci/02070000.NBD
            collectedParameters = []
            collectedUnits = []
            cacheFile = f"{dbdFp.cacheID}.cac"
            openTime = dbdFp.get_fileopen_time()

            self.data['cache'][dbdType] = cacheFile
            self.data['open'][dbdType] = openTime
            # Save the open time of the first timestamp of the available data
            self.data['timestamp'][dbdType] = openTime

            # If there is no data at all, return
            if np.size(dbdData[0]) == 0:
                self.data[dbdType] = dataObj
                self.data['columns'][dbdType] = collectedParameters
                self.data['units'][dbdType] = collectedUnits
                self.data['input'][dbdType] = inputFile
                dbdFp.close()
                return

            timeIdx = dbdFp.parameterNames.index(timeDimension)
            timeLen = len(dbdData[timeIdx][1])

            # Save the open time of the first timestamp of the available data
            if timeLen > 0:
                self.data['timestamp'][dbdType] = dbdData[timeIdx][1][0]

            if not returnDict:
                idx = 0
                for p in dbdFp.parameterNames:
                    # skip columns that are all NaN or [0, Nan, ....]
                    data = dbdData[idx][1]
                    if np.size(data) == 0:
                        idx = idx + 1
                        continue
                    dataLen = len(data)
                    nanLen = len(data[np.isnan(data)])
                    if data[0] == 0.0 and (nanLen + 1) == dataLen:
                        #print("SCREEN1: %s" % (p))
                        idx = idx + 1
                        continue
                    if nanLen == dataLen:
                        #print("SCREEN2: %s" % (p))
                        idx = idx + 1
                        continue

                    # Pad data? This is a bug in dbdreader
                    if len(dbdData[idx][1]) != timeLen:
                        tempVar = np.array([np.nan] * timeLen)
                        tempVar[0:len(dbdData[idx][1])] = dbdData[idx][1]
                        data = tempVar
                        print("  Resized:", dbdFp.parameterNames[idx])
                    dataObj[p] = (("time"), data)
                    #self.stopToDebug()
                    collectedParameters.append(p)
                    collectedUnits.append(dbdFp.parameterUnits[p])
                    idx = idx + 1
            else:
                for p in dbdFp.parameterNames:
                    dataObj[p] = dbdData[idx][1]
                    collectedParameters.append(p)
                    collectedUnits.append(dbdFp.parameterUnits[p])

            # Final assignments into object .data object
            self.data[dbdType] = dataObj
            self.data['columns'][dbdType] = collectedParameters
            self.data['units'][dbdType] = collectedUnits
            self.data['input'][dbdType] = inputFile
            dbdFp.close()

        #self.stopToDebug()

    def readTbd(self):
        '''
        This function reads a glider tbd file using the
        Teledyne Webb linux binary dbd2asc.  This
        also reads the corresponding cache file for additional
        metadata.
        '''

        # Open tbd binary file
        tbdfp = open(self.tbdFile, 'rb')

        # Read ascii section of tbd file
        for i in range(0, 14):
            line = tbdfp.readline()
            lnParts = line.decode('utf-8', 'replace').split(':')
            if lnParts[0] == 'sensors_per_cycle':
                self.data['cacheMetadata']['sensorCount'] = int(lnParts[1])
            if lnParts[0] == 'sensor_list_factored':
                self.data['cacheMetadata']['factored'] = int(lnParts[1])
            if lnParts[0] == 'state_bytes_per_cycle':
                self.data['cacheMetadata']['stByteNum'] = int(lnParts[1])
            if lnParts[0] == 'total_num_sensors':
                self.data['cacheMetadata']['totalSensors'] = int(lnParts[1])
            if lnParts[0] == 'sensor_list_crc':
                self.data['cacheMetadata']['cacheFile'] = str(lnParts[1].strip())

        # Close the tbd file
        tbdfp.close()

        # Scan cache file for timestamp column for decoding in
        # binary.  Also read columns, units and byte sizes for columns.
        # Column order is ASSUMED to be sequential starting at 0 and
        # not skipping integers!

        cacheFile = os.path.join(
            self.cacheDir, '%s.cac' %
            (self.data['cacheMetadata']['cacheFile']))

        if not os.path.isfile(cacheFile):
            tbdfp.close()
            sys.exit('Glider(): FATAL: Glider cache file not found: %s' % (cacheFile))

        with open(cacheFile) as fp:
            ln = fp.readline()
            while ln:
                data = ln.strip().split()
                # Sensors turned on
                if data[1] == 'T':
                    self.data['byteSize'].append(int(data[4]))
                    self.data['columns'].append(data[5])
                    self.data['units'].append(data[6])
                ln = fp.readline()

        # Decode the tbd using the Teledyne Webb linux binary

        # Command to execute
        cmd = [self.dbd2asc, '-c', self.cacheDir, self.tbdFile]

        processOutput = subprocess.run(cmd, stdout=subprocess.PIPE)
        returnCode = processOutput.returncode

        if returnCode != 0:
            if self.debugFlag:
                print("DEBUG: dbd2asc failed to run.  Debug is turned on.  This program will now pause in the debugger.")
                print("This program will exit upon continuing the program from the debugger.")

                self.stopToDebug()

            sys.exit('Glider(): FATAL: dbd2asc failed to run')

        output = io.StringIO(processOutput.stdout.decode())

        # Process stdout
        dbdData = []
        hdrFlag = True
        while True:
            ln = output.readline()
            if not ln:
                break
            ln = ln.strip()
            data = ln.split()
            dataLen = len(data)
            # Ignore header information until we encounter data below
            if hdrFlag:
                if dataLen == self.data['cacheMetadata']['sensorCount']:
                    # The first row of this type are the columns, the next two
                    # rows should be units and byte sizes.  Read those and
                    # set the header flag to False.
                    ln = output.readline()
                    ln = output.readline()
                    hdrFlag = False
            else:
                dbdData.append(data)

        # Store the data portion of the tbd file
        self.data['asc'] = np.array(dbdData).astype(float)

    def readSbd(self):
        '''
        This function reads a glider sbd file.  This
        also reads the corresponding cache file for additional
        metadata.
        '''

        # Open sbd binary file
        sbdfp = open(self.sbdFile, 'rb')

        # Read ascii section of tbd file
        for i in range(0, 14):
            line = sbdfp.readline()
            lnParts = line.decode('utf-8', 'replace').split(':')
            if lnParts[0] == 'sensors_per_cycle':
                self.data['sbdMetadata']['sensorCount'] = int(lnParts[1])
            if lnParts[0] == 'sensor_list_factored':
                self.data['sbdMetadata']['factored'] = int(lnParts[1])
            if lnParts[0] == 'state_bytes_per_cycle':
                self.data['sbdMetadata']['stByteNum'] = int(lnParts[1])
            if lnParts[0] == 'total_num_sensors':
                self.data['sbdMetadata']['totalSensors'] = int(lnParts[1])
            if lnParts[0] == 'sensor_list_crc':
                self.data['sbdMetadata']['cacheFile'] = str(lnParts[1].strip())

        # Close the sbd file
        sbdfp.close()

        # Scan cache file for timestamp column for decoding in
        # binary.  Also read columns, units and byte sizes for columns.
        # Column order is ASSUMED to be sequential starting at 0 and
        # not skipping integers!

        cacheFile = os.path.join(
            self.cacheDir, '%s.cac' %
            (self.data['sbdMetadata']['cacheFile']))

        if not os.path.isfile(cacheFile):
            sys.exit('Glider(): FATAL: Glider cache file not found: %s' % (cacheFile))

        with open(cacheFile) as fp:
            ln = fp.readline()
            while ln:
                data = ln.strip().split()
                # Sensors turned on
                if data[1] == 'T':
                    self.data['sbdbyteSize'].append(int(data[4]))
                    self.data['sbdcolumns'].append(data[5])
                    self.data['sbdunits'].append(data[6])
                ln = fp.readline()

        # Decode the sbd using the Teledyne Webb linux binary

        # Command to execute
        cmd = [self.dbd2asc, '-c', self.cacheDir, self.sbdFile]

        processOutput = subprocess.run(cmd, stdout=subprocess.PIPE)
        returnCode = processOutput.returncode

        if returnCode != 0:
            if self.debugFlag:
                print("DEBUG: dbd2asc failed to run.  Debug is turned on.  This program will now pause in the debugger.")
                print("This program will exit upon continuing the program from the debugger.")

                self.stopToDebug()

            sys.exit('Glider(): FATAL: dbd2asc failed to run')

        output = io.StringIO(processOutput.stdout.decode())

        # Process stdout
        dbdData = []
        hdrFlag = True
        while True:
            ln = output.readline()
            if not ln:
                break
            ln = ln.strip()
            data = ln.split()
            dataLen = len(data)
            # Ignore header information until we encounter data below
            if hdrFlag:
                if dataLen == self.data['sbdMetadata']['sensorCount']:
                    # The first row of this type are the columns, the next two
                    # rows should be units and byte sizes.  Read those and
                    # set the header flag to False.
                    ln = output.readline()
                    ln = output.readline()
                    hdrFlag = False
            else:
                dbdData.append(data)

        # Store the data portion of the sbd file
        self.data['sbd'] = np.array(dbdData).astype(float)

    def readEchogram(self):
        '''
        This function reads the glider tbd file and extracts the embedded
        echogram from the echometrics data.  The `dbd2asc` file cannot be used
        since it truncates the least significant bits in which the echogram is
        embedded.

        NOTE: This function `readEchogram` should only be used for
        extracting encoded echogram information embedded in the echometrics
        data.  All other glider files may be read using `dbd2asc`.  We highly
        recommend using the python `dbdreader` module.

        This function automatically determines if the glider
        was in "egram" or "combo" mode.  Prior knowledge of the
        operational mode is not necessary.
        '''

        # Obtain timestamp and water depth (bar)
        # For demonstration only
        #a = self.extractColumns('asc', columns=['sci_m_present_time','sci_water_pressure'],
        #    ignoreNaNColumns=['sci_water_pressure'])

        # Obtain timestamp and water depth (bar) as a dictionary
        # For demonstration only
        #t = self.extractColumns('asc', columns=['sci_m_present_time','sci_water_pressure'],
        #    ignoreNaNColumns=['sci_water_pressure'], asDict=True)

        # Obtain tbd values by direct read rather than through dbd2asc.  It has been
        # found that significant digits are lost when using dbd2asc.
        # This code is formerly: dbDecode3.py

        # Method 1

        '''
        bdFile = open(self.tbdFile, 'rb')

        # Read the first 14 records
        for i in range(0,14):
            line = bdFile.readline()

        # If a sensor list is included in the tbd, skip those lines
        if self.data['cacheMetadata']['factored'] == 0:
            for i in range(0, self.data['cacheMetadata']['totalSensors']):
                line = bdFile.readline()

        # Populate the repeat value cache with zeros
        repeatVal = [0] * self.data['cacheMetadata']['sensorCount']

        stByteNum = self.data['cacheMetadata']['stByteNum']
        timestampColumn = self.data['columns'].index('sci_m_present_time')

        ch = ' '
        numRecords = 0
        while True:
            #read off state bits
            # For python3, use decode to convert bytes to string
            try:
                chRead = bdFile.read(1)
                ch = chRead.decode('utf-8','replace')
            except:
                print("Glider.readEchogram(): Binary read error.")
                sys.exit()

            if not ch:
                break

            # A state bit of "d" means data
            if ch == 'd':
                #print("State bit:",ch)
                #bitmask to shift and look at 2bit state / sensor
                mask = 3      #binary 11000000
                stBits = bdFile.read(stByteNum)
                c2 = 0
                setFlag = 0
                count = 0
                dataRec = np.empty(0, int)
                for rval in stBits:
                    # Python 2 way of unpacking bytes
                    #[val] = struct.unpack('B', rval)
                    # For python3, pulling a value out of stBits using a
                    # for loop converts the byte to an integer as was
                    # done for python 2 using unpack above.
                    val = rval
                    for i in range(0,8,2):
                        # Repeat value from last record
                        if (val>>(6-i)) & mask == 1:
                            if count < 7 or count == 17:
                                setFlag = 1
                                #print(str(repeatVal[count]) + ' ', end='')
                                dataRec = np.append(dataRec, repeatVal[count])
                        # Read a new value and store in case of repeat value
                        if (val>>(6-i)) & mask == 2:
                            #bob
                            if count < 7:
                                setFlag = 1
                                [sensorVal] = struct.unpack('>i', bdFile.read(4))
                                #print(str(sensorVal) + ' ', end='')
                                dataRec = np.append(dataRec, sensorVal)
                                c2 += 4
                                repeatVal[count] = sensorVal #in case next cycle sensor updated but didnt send do to repeat value
                                #print(' ', end='')
                            elif count == timestampColumn:
                                [sensorVal] = struct.unpack('>d', bdFile.read(8))
                                c2 += 8
                                repeatVal[count] = sensorVal
                                if setFlag == 1:
                                    #print(sensorVal, end='')
                                    dataRec = np.append(dataRec, sensorVal)
                                    #print(' ', end='')
                            else:
                                bdFile.read(4)  #read off unwanted sensor
                                c2 += 4
                        count = count + 1
                if setFlag:
                    #print('')
                    #self.stopToDebug()
                    if numRecords == 0:
                        self.data['echogram'] = np.column_stack(dataRec)
                    else:
                        self.data['echogram'] = np.append(self.data['echogram'], np.column_stack(dataRec), axis=0)
                    numRecords = numRecords + 1

        bdFile.close()
        '''

        # Method 2
        # https://github.com/hstats/Glider-echo-tools
        # The transmission order is important for the data bins:
        #   metrics_list = [echometrics.depth_integral,
        #           echometrics.proportion_occupied,
        #           echometrics.aggregation_index,
        #           echometrics.sv_avg,
        #           echometrics.center_of_mass,
        #           echometrics.inertia,
        #           echometrics.equivalent_area]
        # DBD columns are:
        #   NOTE: The order obtained from the DBD is not guaranteed!
        dataOrder = [
            'sci_echodroid_aggindex', 'sci_echodroid_ctrmass',
            'sci_echodroid_eqarea', 'sci_echodroid_inertia',
            'sci_echodroid_propocc', 'sci_echodroid_sa', 'sci_echodroid_sv']

        #dbd = dbdreader.DBD(self.tbdFile, cacheDir=self.cacheDir)

        # The echogram is contained within the tdb file (during deployment and
        # transmitted via iridium).  When the glider data is obtained, the pseduogram
        # will be also available in the ebd file.
        # If the tbd or ebd does not exist, return.
        echogram_source = None

        if 'tbd' in self.data.keys():
            echogram_source = 'tbd'

        # If both types are available, prefer the ebd file
        if 'ebd' in self.data.keys():
            echogram_source = 'ebd'

        if self.debugFlag:
            print("Echogram source:", echogram_source)

        if not echogram_source:
            self.data['echogram'] = None
            return

        self.data['echogram_source'] = echogram_source
        self.data['echogram_source_file'] = self.data['input'][echogram_source]

        # Repack floats as integers for decoding below
        # We have to pre-pad a 0 to match Method 1
        self.data['psu'] = None

        tmVar = self.findTimeVariable(self.data['columns'][echogram_source])
        if self.debugFlag:
            print("Time variable:", tmVar)
        if tmVar is None:
            self.data['echogram'] = None
            return

        #self.stopToDebug()

        # Convert float data to integer for processing (if egram is embedded)
        tmData = self.data[echogram_source][tmVar]
        nanMask = None
        for idx in dataOrder:
            try:
                dbdData = self.data[echogram_source][idx]
            except Exception:
                self.data['echogram'] = None
                return

            #self.stopToDebug()
            # Obtain a NaN mask for the echogram.  For gliders, the acoustics can
            # be turned on and off at different points in the dive.
            if nanMask is None:
                nanMask = dbdData.isnull()
            # Drop NaN data
            dbdData = dbdData.where(nanMask == False, drop=True)  # noqa: E712
            tmpData = [0]
            #self.stopToDebug()
            for v in [struct.unpack('>i', struct.pack('>f', v)) for v in dbdData]:
                tmpData.append(v[0])
            #self.stopToDebug()
            dbdData = tmpData
            if self.data['psu'] is None:
                self.data['psu'] = np.row_stack(dbdData)
            else:
                self.data['psu'] = np.append(self.data['psu'], np.row_stack(dbdData), axis=1)

        # Drop time elements that are part of the NaN mask
        tmData = tmData.where(nanMask == False, drop=True)  # noqa: E712
        # Pad a time element to keep the data size consistent
        tmData = np.insert([0.0], 1, tmData)

        #self.stopToDebug()
        # Concatenate columns of data together for processing below
        self.data['psu'] = np.append(self.data['psu'], np.row_stack(tmData), axis=1)

        # Use method 2 for echogram
        self.data['echogram'] = self.data['psu']

        # Perform reorder step
        (numRows, numCols) = self.data['echogram'].shape

        # Density (Sv) (dB) bins are categorized 0 to 7
        #bins = {0: '-5.0', 1:'-15.0', 2:'-22.5', 3:'-27.5', 4:'-32.5', 5:'-40.0', 6:'-50.0', 7:'-60.0' }
        if 'vbsBins' in self.args:
            vbs = np.array([np.nan] * 15)
            vbs[1::2] = self.args['vbsBins']
            not_nan = np.logical_not(np.isnan(vbs))
            indices = np.arange(len(vbs))
            vbsNew = np.interp(indices, indices[not_nan], vbs[not_nan])
            # Fill in the edge points with proper values
            vbsNew[0] = vbsNew[0] - (vbsNew[2] - vbsNew[1])
            vbsNew[-1] = vbsNew[-1] + (vbsNew[-2] - vbsNew[-3])
            # Save bin assignments
            bins = {}
            for i in range(0, 8):
                bins[i] = vbsNew[i * 2]
        else:
            # Default bins (if not defined)
            bins = {0: -5.0, 1: -15.0, 2: -22.5, 3: -27.5, 4: -32.5, 5: -40.0, 6: -50.0, 7: -60.0 }

        self.data['echogram_bins'] = bins
        self.data['echogram_mode'] = None

        # Data is hidden in the last three bits of data sent over iridium
        #mask of 7(binary 111) for 3bit vals
        mask = 7

        # No data is numRows = 1
        if numRows == 1:
            self.data['echogram'] = None
            return

        #self.stopToDebug()

        # reord.py reads off the first row to skip it
        # Collect the bit values 0 to 7 first to allow for
        # validation of routine.
        fullEchogram = None
        for row in range(1, numRows):
            words = self.data['echogram'][row, :]
            if len(words) != 0:
                #print(words)
                # Skip apparent missing data
                if np.all(words[0:7] == [0, 0, 0, 0, 0, 0, 0]) is True:
                    continue
                if int(words[2]) == -1032813281:
                    self.data['echogram_mode'] = "egram"
                    # This decodes "egram" mode
                    # time
                    #print(words[7] + ' ', end=''),
                    #print("%f " % (words[7]), end=''),
                    decodedEchogram = np.array(words[7])
                    #decodedEchogram = np.array(words[7])
                    #2 32bit vals two high bits are left over, read off 3 at time
                    for i in range(0, 30, 3):
                        #print(str((int(words[6])>>i) & mask) + ' ', end=''),
                        #print(bins[(int(words[6])>>i) & mask] + ' ', end=''),
                        #print("%s " % (bins[(int(words[6])>>i) & mask]), end=''),
                        #decodedEchogram = np.append(decodedEchogram, (bins[(int(words[6])>>i) & mask]))
                        decodedEchogram = np.append(decodedEchogram, ((int(words[6]) >> i) & mask))
                    for i in range(0, 30, 3):
                        #print(str((int(words[3])>>i) & mask) + ' ', end=''),
                        #print(bins[(int(words[3])>>i) & mask] + ' ', end=''),
                        #print("%s " % (bins[(int(words[3])>>i) & mask]), end=''),
                        #decodedEchogram = np.append(decodedEchogram, (bins[(int(words[3])>>i) & mask]))
                        decodedEchogram = np.append(decodedEchogram, ((int(words[4]) >> i) & mask))
                    #print('\n')
                    '''
                    if fullEchogram is None:
                        fullEchogram = np.column_stack(decodedEchogram)
                    else:
                        fullEchogram = np.append(fullEchogram, np.column_stack(decodedEchogram), axis=0)
                    '''
                    #time
                    #print(words[7] + ' ', end=''),
                    #print("%f " % (words[7]), end=''),
                    #decodedEchogram = np.array(words[7])
                    #2 32bit vals two high bits are left over, read off 3 at time
                    for i in range(0, 30, 3):
                        #print(str((int(words[0])>>i) & mask) + ' ', end=''),
                        #print(bins[(int(words[0])>>i) & mask] + ' ', end=''),
                        #print("%s " % (bins[(int(words[0])>>i) & mask]), end=''),
                        #decodedEchogram = np.append(decodedEchogram, (bins[(int(words[0])>>i) & mask]))
                        decodedEchogram = np.append(decodedEchogram, ((int(words[0]) >> i) & mask))
                    for i in range(0, 30, 3):
                        #print(str((int(words[5])>>i) & mask) + ' ', end=''),
                        #print(bins[(int(words[5])>>i) & mask] + ' ', end=''),
                        #print("%s " % (bins[(int(words[5])>>i) & mask]), end=''),
                        #decodedEchogram = np.append(decodedEchogram, (bins[(int(words[5])>>i) & mask]))
                        decodedEchogram = np.append(decodedEchogram, ((int(words[5]) >> i) & mask))
                    #print('\n')
                    '''
                    if fullEchogram is None:
                        fullEchogram = np.column_stack(decodedEchogram)
                    else:
                        fullEchogram = np.append(fullEchogram, np.column_stack(decodedEchogram), axis=0)
                    '''
                    #time
                    #print(words[7] + ' ', end=''),
                    #print("%f " % (words[7]), end=''),
                    #decodedEchogram = np.array(words[7])
                    #2 32bit vals two high bits are left over, read off 3 at time
                    for i in range(0, 30, 3):
                        #print(str((int(words[1])>>i) & mask) + ' ', end=''),
                        #print(bins[(int(words[1])>>i) & mask] + ' ', end=''),
                        #print("%s " % (bins[(int(words[1])>>i) & mask]), end=''),
                        #decodedEchogram = np.append(decodedEchogram, (bins[(int(words[1])>>i) & mask]))
                        decodedEchogram = np.append(decodedEchogram, ((int(words[1]) >> i) & mask))
                    for i in range(0, 30, 3):
                        #print(str((int(words[4])>>i) & mask) + ' ', end=''),
                        #print(bins[(int(words[4])>>i) & mask] + ' ', end=''),
                        #print("%s " % (bins[(int(words[4])>>i) & mask]), end=''),
                        #decodedEchogram = np.append(decodedEchogram, (bins[(int(words[4])>>i) & mask]))
                        decodedEchogram = np.append(decodedEchogram, ((int(words[3]) >> i) & mask))
                    #print('\n')
                    if fullEchogram is None:
                        fullEchogram = np.column_stack(decodedEchogram)
                    else:
                        fullEchogram = np.append(fullEchogram, np.column_stack(decodedEchogram), axis=0)
                else:
                    # This decodes combo mode
                    self.data['echogram_mode'] = "combo"
                    #print('combo')
                    #time
                    #print("%f " % (words[7]), end=''),
                    decodedEchogram = np.array(words[7])
                    #print(words[7] + ' ', end=''),
                    #2 32bit vals are metrics with LS 9bits used for 3 layer values
                    for i in range(0, 9, 3):
                        #print(str((int(words[6])>>i) & mask) + ' ', end=''),
                        #print(bins[(int(words[6])>>i) & mask] + ' ', end=''),
                        #print("%s " % (bins[(int(words[6])>>i) & mask]), end=''),
                        #decodedEchogram = np.append(decodedEchogram, (bins[(int(words[6])>>i) & mask]))
                        decodedEchogram = np.append(decodedEchogram, ((int(words[6]) >> i) & mask))
                    for i in range(0, 9, 3):
                        #print(str((int(words[4])>>i) & mask) + ' ', end=''),
                        #print(bins[(int(words[4])>>i) & mask] + ' ', end=''),
                        #print("%s " % (bins[(int(words[4])>>i) & mask]), end=''),
                        #decodedEchogram = np.append(decodedEchogram, (bins[(int(words[4])>>i) & mask]))
                        decodedEchogram = np.append(decodedEchogram, ((int(words[4]) >> i) & mask))
                    for i in range(0, 9, 3):
                        #print(str((int(words[0])>>i) & mask) + ' ', end=''),
                        #print(bins[(int(words[0])>>i) & mask] + ' ', end=''),
                        #print("%s " % (bins[(int(words[0])>>i) & mask]), end=''),
                        #decodedEchogram = np.append(decodedEchogram, (bins[(int(words[0])>>i) & mask]))
                        decodedEchogram = np.append(decodedEchogram, ((int(words[0]) >> i) & mask))
                    for i in range(0, 9, 3):
                        #print(str((int(words[5])>>i) & mask) + ' ', end=''),
                        #print(bins[(int(words[5])>>i) & mask] + ' ', end=''),
                        #print("%s " % (bins[(int(words[5])>>i) & mask]), end=''),
                        #decodedEchogram = np.append(decodedEchogram, (bins[(int(words[5])>>i) & mask]))
                        decodedEchogram = np.append(decodedEchogram, ((int(words[5]) >> i) & mask))
                    for i in range(0, 9, 3):
                        #print(str((int(words[1])>>i) & mask) + ' ', end=''),
                        #print(bins[(int(words[1])>>i) & mask] + ' ', end=''),
                        #print("%s " % (bins[(int(words[1])>>i) & mask]), end=''),
                        decodedEchogram = np.append(decodedEchogram, ((int(words[1]) >> i) & mask))
                    for i in range(0, 9, 3):
                        #print(str((int(words[3])>>i) & mask) + ' ', end=''),
                        #print(bins[(int(words[3])>>i) & mask] + ' ', end=''),
                        #print("%s " % (bins[(int(words[3])>>i) & mask]), end=''),
                        decodedEchogram = np.append(decodedEchogram, ((int(words[3]) >> i) & mask))
                    for i in range(0, 6, 3):
                        #print(str((int(words[2])>>i) & mask) + ' ', end=''),
                        #print(bins[(int(words[2])>>i) & mask] + ' ', end=''),
                        #print("%s " % (bins[(int(words[2])>>i) & mask]), end=''),
                        decodedEchogram = np.append(decodedEchogram, ((int(words[2]) >> i) & mask))
                    #print('\n')
                    #self.stopToDebug()
                    # Temporary kludge to mask the -5.0 bins at the end
                    #if decodedEchogram[21] == '-5.0':
                    #    decodedEchogram[21] = '-60.0'
                    if fullEchogram is None:
                        fullEchogram = np.column_stack(decodedEchogram)
                    else:
                        fullEchogram = np.append(fullEchogram, np.column_stack(decodedEchogram), axis=0)

        # What type of echogram do we have here?
        # Make the time adjustment appropriate for the mode the glider
        # is in.
        if self.data['echogram_mode']:
            if self.data['echogram_mode'] == 'combo':
                # In combo mode, the VBS scan is #10 for a given time at VBS scan #30
                # Offset is -20.0 seconds
                fullEchogram[:, 0] = fullEchogram[:, 0] - 20.0
            if self.data['echogram_mode'] == 'egram':
                # In egram mode, the time is for VBS scan #30, the three scans of information
                # are VBS scan #0 (-30.0 seconds); VBS scan #10 (-20.0 seconds)
                # and VBS scan #20 (-10.0 seconds)

                # Reshape echogram and adjust time

                # Initialize row 0
                # Grab the time of the metric recording
                techo = fullEchogram[0, 0]
                reshapeEchogram = fullEchogram[0, 0:21]
                reshapeEchogram = reshapeEchogram.reshape(-1, reshapeEchogram.shape[0])
                reshapeEchogram[0, 0] = techo - 30.0

                # Row 1
                rec = np.append([techo - 20.0], fullEchogram[0, 21:41])
                reshapeEchogram = np.append(reshapeEchogram, np.column_stack(rec), axis=0)

                # Row 2
                rec = np.append([techo - 10.0], fullEchogram[0, 41:61])
                reshapeEchogram = np.append(reshapeEchogram, np.column_stack(rec), axis=0)

                # Perform the same to the remaining rows
                for z in range(1, fullEchogram.shape[0]):
                    # Grab the time reading
                    techo = fullEchogram[z, 0]
                    for s in range(0, 3):
                        rec = np.append(
                            [techo - ((3 - s) * 10.0)],
                            fullEchogram[z, (s * 20) + 1: ((s + 1) * 20) + 1])
                        #breakpoint()
                        reshapeEchogram = np.append(reshapeEchogram, np.column_stack(rec), axis=0)

                fullEchogram = reshapeEchogram

        # Check for reprocess and test pattern arguments here
        process = self.args.get('reprocess', None)
        testPattern = self.args.get('testPattern', None)

        # Copy the bits before we change them
        if fullEchogram is None:
            bitsEchogram = None
        else:
            # Apply --testPattern and/or --process options here
            # If --testPattern is set then change the bits to a test pattern.
            # Don't do anything if --process is also set, the test pattern will
            # already be applied.
            if testPattern and not process:
                # Replace echogram with 0-7 bit values in sequence
                bitValue = -1.0
                for i in range(0, fullEchogram.shape[0]):
                    bitValue = bitValue + 1.0
                    if bitValue >= 8.0:
                        bitValue = 0.0
                    fullEchogram[i, 1:21] = [bitValue] * 20

            bitsEchogram = fullEchogram.copy()

            # Convert echogram from bits to values
            #https://stackoverflow.com/questions/62864282/numpy-apply-function-to-every-item-in-array
            #apply_thresholds = lambda b: bins[b]
            def apply_thresholds(b):
                return bins[b]
            #fullEchogram[:,1:] = np.stack(np.vectorize(apply_thresholds)(fullEchogram[:,1:]), axis=1).T
            fullEchogram[:, 1:] = np.stack(np.vectorize(apply_thresholds)(bitsEchogram[:, 1:]), axis=1).T

        # Return the final echogram data
        #if not(fullEchogram is None):
        #    fullEchogram = fullEchogram.astype(float)
        self.data['echogram'] = fullEchogram
        self.data['echogram_bits'] = bitsEchogram

        # If the reprocess flag is set, at this point we only search for valid VBS data
        # and place it in self.data['echogram_*'].  An echogram also should be detected
        # in the data stream.
        if process and np.all(fullEchogram):
            self.findRawVBS(fullEchogram, process)

    def getRawVBSFilename(self, process):
        '''
        Determine raw VBS filename

        NOTE: Pull out the output types here too
        '''

        # If configuration file was not found, return
        if not self.echotools:
            return None

        # default search path is {deployments}/{glider}/{date}/echotools/eit/{data}.vbs
        # otherwise specified by {deployments}/{glider}/{date}/{raw_vbs}

        # Look for a reprocess configuration
        if self.echotools.get('echotools', None):
            # Obtain deployment path
            deploymentPath = self.echotools['echotools']['paths']['deployments']
            # Get deployment name and glider name
            deployment = self.echotools['echotools'].get('deployment_name', None)
            glider = self.echotools['echotools'].get('teledyne_webb_vehicle_name', None)
            if self.echotools['echotools'].get('reprocessing', None):
                if self.echotools['echotools']['reprocessing'].get(process, None):
                    processConfig = self.echotools['echotools']['reprocessing'][process]
                    self.echotools['processConfig'] = processConfig

        # Assemble path
        fname = os.path.join(deploymentPath, glider, deployment, processConfig['raw_vbs'])
        if os.path.isfile(fname):
            return fname

        return None

    def findRawVBS(self, fullEchogram, process):
        '''
        Search for the raw VBS from the specified vbs file.
        '''

        vbsFilename = self.getRawVBSFilename(process)
        processConfig = self.echotools['processConfig']
        outputs = processConfig['output']
        bins = processConfig.get('svb_thresholds', None)

        # Create the unbin values
        vbs = np.array([np.nan] * 15)
        vbs[1::2] = bins
        not_nan = np.logical_not(np.isnan(vbs))
        indices = np.arange(len(vbs))
        vbsNew = np.interp(indices, indices[not_nan], vbs[not_nan])
        # Fill in the edge points with proper values
        vbsNew[0] = vbsNew[0] - (vbsNew[2] - vbsNew[1])
        vbsNew[-1] = vbsNew[-1] + (vbsNew[-2] - vbsNew[-3])
        # Save bin assignments
        unbins = {}
        for i in range(0, 8):
            unbins[i] = vbsNew[i * 2]

        #apply_thresholds = lambda b: unbins[b]
        def apply_thresholds(b):
            return unbins[b]

        # Thresholds must be set even if they are the same as the deployment!
        if not bins:
            return

        # No filename, return
        if not vbsFilename:
            return

        # Hunt for the first matching profile => fullEchogram[timeDim, 0]
        # Should be approximately 30 sec prior
        tsMetric = fullEchogram[0, 0] - 30.0
        vbsFP = open(vbsFilename, 'r')
        tsVbs = 0.0

        success = True
        while tsVbs < tsMetric:
            ln = vbsFP.readline()
            ln = ln.strip()
            if ln:
                data = ln.split(" ")
                tsVbs = float(data[2])
            else:
                success = False
                break

        # Sync failed
        if not success:
            return

        #print(tsMetric, data)
        # Now we can read the groups of 30, if we encounter a problem
        # we flag it as a QC problem and continue.

        endOfFile = False
        ct = 0
        for t in range(0, fullEchogram.shape[0]):
            tsMetric = fullEchogram[t, 0]
            vbsData = []

            while tsVbs < tsMetric:
                vbsArray = data[5].split(',')
                nbins = int(vbsArray[5])
                vbsValues = [tsVbs]
                vbsValues = vbsValues + [float(vbsArray[6 + i]) for i in range(0, nbins)]
                vbsData.append(vbsValues)

                ln = vbsFP.readline()
                ln = ln.strip()
                if ln:
                    data = ln.split(" ")
                    tsVbs = float(data[2])
                else:
                    # Unexpected end of file
                    endOfFile = True
                    break
                # If we have read 30 values, assume we are done
                # as well.
                if len(vbsData) == 30:
                    break

            # We do not need to generate the metrics, only the digitized bits
            # according to the thresholds provided! Skip any data block that
            # does not contain 30 scans.

            echodata = np.array(vbsData)
            #print(echodata.shape)
            if echodata.shape[0] == 30:
                # Determine output types to create
                for outType in outputs:
                    if outType == "vbs":
                        if ct == 0:
                            #self.data['echogram_vbs'] = np.column_stack(echodata)
                            self.data['echogram_vbs'] = echodata
                        else:
                            self.data['echogram_vbs'] = np.append(self.data['echogram_vbs'], echodata, axis=0)
                            #breakpoint()

                    if outType == "combo":
                        digitized = np.digitize(echodata[10][1:], bins)
                        norm = np.vectorize(apply_thresholds)(digitized)
                        newdata = np.append(echodata[10][0], norm)
                        if ct == 0:
                            self.data['echogram_combo'] = np.column_stack(newdata)
                        else:
                            self.data['echogram_combo'] = np.append(self.data['echogram_combo'], np.column_stack(newdata), axis=0)

                    if outType == "egram":
                        digitized = np.digitize(echodata[0][1:], bins)
                        norm = np.vectorize(apply_thresholds)(digitized)
                        newdata = np.append(echodata[0][0], norm)
                        if ct == 0:
                            self.data['echogram_egram'] = np.column_stack(newdata)
                        else:
                            self.data['echogram_egram'] = np.append(self.data['echogram_egram'], np.column_stack(newdata), axis=0)

                        digitized = np.digitize(echodata[10][1:], bins)
                        norm = np.vectorize(apply_thresholds)(digitized)
                        newdata = np.append(echodata[10][0], norm)
                        self.data['echogram_egram'] = np.append(self.data['echogram_egram'], np.column_stack(newdata), axis=0)

                        digitized = np.digitize(echodata[20][1:], bins)
                        norm = np.vectorize(apply_thresholds)(digitized)
                        newdata = np.append(echodata[20][0], norm)
                        self.data['echogram_egram'] = np.append(self.data['echogram_egram'], np.column_stack(newdata), axis=0)
                ct = ct + 1

            if endOfFile:
                break

    def applyDeploymentGlobalMetadata(self, ncDS):
        '''
        This function applies the deployment.json global metadata to the xarray Dataset object.
        '''

        # Add the 'attributes' items as global metadata.
        for dictKey in self.deployment['attributes'].keys():
            ncDS.attrs[dictKey] = self.deployment['attributes'][dictKey]

        return ncDS

    def applyTemplateGlobalMetadata(self, ncDS):
        '''
        This function applies the {template}.json global metadata to the xarray Dataset object.
        '''

        # Add the 'attributes' items as global metadata.
        #self.stopToDebug()
        for dictKey in self.template['attributes'].keys():
            ncDS.attrs[dictKey] = self.template['attributes'][dictKey]

        return ncDS

    def readTemplateUnlimitedDims(self):
        '''
        This function reads {template}.json and allocates unlimited dimensions for the xarray Dataset object.
        '''

        # Look for unlimited dimensions
        self.ncUnlimitedDims = []
        if 'dimensions' in self.template.keys():
            for dictKey in self.template['dimensions'].keys():
                dimVal = self.template['dimensions'][dictKey]
                # Dimensions with -1 are unlimited
                if dimVal == -1:
                    self.ncUnlimitedDims.append(dictKey)

    def obtainVariableName(self, sensorName):
        '''
        This function obtains a variable name from a given glider sensor
        name.  There is sometimes some mapping.  This requires use of
        the template file and checking 'rename_from' elements.
        '''

        # If there is a remapping in the template, return that name
        for vkey in self.template['variables'].keys():
            if 'rename_from' in self.template['variables'][vkey].keys():
                if sensorName in self.template['variables'][vkey]['rename_from']:
                    return vkey

        # Otherwise return the sensor name
        return sensorName

    def obtainFillValue(self, varName):
        '''
        This function obtains a _FillValue to use for the specified variable.
        If a fill value is not configured, the default is -9999.9.
        '''

        fillValue = -9999.9
        fillValueDtype = 'double'

        # Hunt for fill value in template
        vkeys = self.template['variables'].keys()
        if varName in vkeys:
            if 'attributes' in self.template['variables'][varName].keys():
                if '_FillValue' in self.template['variables'][varName]['attributes'].keys():
                    fillValue = self.template['variables'][varName]['attributes']['_FillValue']['data']
                    fillValueDtype = self.template['variables'][varName]['attributes']['_FillValue']['type']

        return (fillValue, fillValueDtype)

    def addAttributes(self, ncDS, varName):
        '''
        This function adds variable attributes to the netCDF variable from
        the specified template file.
        '''

        vkeys = self.template['variables'].keys()
        if varName in vkeys:
            if 'attributes' in self.template['variables'][varName].keys():
                for akey in self.template['variables'][varName]['attributes'].keys():
                    if akey != '_FillValue':
                        # If the key value is a dict()
                        if hasattr(self.template['variables'][varName]['attributes'][akey], 'keys'):
                            ncDS[varName].attrs[akey] = self.template['variables'][varName]['attributes'][akey]['data']
                        else:
                            ncDS[varName].attrs[akey] = self.template['variables'][varName]['attributes'][akey]

    def collectTbdVariables(self, ncDS):
        '''
        This function collects tbd file variables.
        '''
        #idxTime = self.data['columns']['tbd'].index('sci_m_present_time')
        sensorTimes = self.data['tbd']['sci_m_present_time']

        # Map sensor times to the new dimension times
        # This slow process only needs to be done once (skip if
        # we are storing this separately)
        '''
        if not(self.args.ncSeparate):
            timeDimList = list(ncDS['time'].values)
            newTimeIdx = []
            for sTime in sensorTimes:
                iSel = ncDS['time'].sel(time=sTime)
                if iSel.size == 1:
                    itime = timeDimList.index(iSel)
                else:
                    itime = timeDimList.index(iSel[0])
                newTimeIdx.append(itime)
        else:
        '''
        # TODO: Convert epoch from 1970 to 1990
        newTimeIdx = range(0, len(sensorTimes))
        ncDS['time'] = (("time"), sensorTimes.data)
        ncDS['time'].attrs['units'] = "seconds since 1970-01-01"
        self.fillValues = {}
        self.fillValues['time'] = {'_FillValue': -9999.9, 'dtype': 'double'}
        timeDimList = list(ncDS['time'].values)

        # Add source filename to metadata
        ncDS.attrs['source_file'] = self.data['input']['tbd']

        for sensor in self.template['variables'].keys():
            sensorName = sensor
            if sensorName in self.data['columns']['tbd']:
                #idx = self.data['columns']['tbd'].index(sensorName)
                sensorData = self.data['tbd'][sensorName]

                # Create empty array
                varName = self.obtainVariableName(sensorName)
                (fillValue, fillValueDtype) = self.obtainFillValue(varName)
                varData = np.full(len(timeDimList), fillValue)
                self.fillValues[varName] = {'_FillValue': fillValue, 'dtype': fillValueDtype}

                # Map sensor data to time dimension
                varData[newTimeIdx] = sensorData

                ncDS[varName] = (("time"), varData)

                # Add variable attributes
                self.addAttributes(ncDS, varName)

        return ncDS

    def collectDbdVariables(self, ncDS):
        '''
        This function collects dbd file variables.
        '''
        #idxTime = self.data['columns']['dbd'].index('m_present_time')
        sensorTimes = self.data['dbd']['m_present_time']

        # Map sensor times to the new dimension times
        # This slow process only needs to be done once (skip if
        # we are storing this separately)
        '''
        if not(self.args.ncSeparate):
            timeDimList = list(ncDS['time'].values)
            newTimeIdx = []
            for sTime in sensorTimes:
                iSel = ncDS['time'].sel(time=sTime)
                if iSel.size == 1:
                    itime = timeDimList.index(iSel)
                else:
                    itime = timeDimList.index(iSel[0])
                newTimeIdx.append(itime)
        else:
        '''
        # TODO: Convert epoch from 1970 to 1990
        newTimeIdx = range(0, len(sensorTimes))
        ncDS['time'] = (("time"), sensorTimes.data)
        ncDS['time'].attrs['units'] = "seconds since 1970-01-01"
        self.fillValues = {}
        self.fillValues['time'] = {'_FillValue': -9999.9, 'dtype': 'double'}
        timeDimList = list(ncDS['time'].values)

        # Add source filename to metadata
        ncDS.attrs['source_file'] = self.data['input']['dbd']

        for sensor in self.template['variables'].keys():
            sensorName = sensor
            if sensorName in self.data['columns']['dbd']:
                #idx = self.data['columns']['dbd'].index(sensorName)
                sensorData = self.data['dbd'][sensorName]

                # Create empty array
                varName = self.obtainVariableName(sensorName)
                (fillValue, fillValueDtype) = self.obtainFillValue(varName)
                varData = np.full(len(timeDimList), fillValue)
                self.fillValues[varName] = {'_FillValue': fillValue, 'dtype': fillValueDtype}

                # Map sensor data to time dimension
                varData[newTimeIdx] = sensorData

                ncDS[varName] = (("time"), varData)

                # Add variable attributes
                self.addAttributes(ncDS, varName)

        return ncDS

    def getScale(self, sensorName):
        '''
        Check the template for a scaling factor prior to assigning the
        final data.
        '''

        scaleFactor = None

        vkeys = self.template['variables'].keys()
        if sensorName in vkeys:
            if 'scale' in self.template['variables'][sensorName].keys():
                scaleFactor = self.template['variables'][sensorName]['scale']

        return scaleFactor

    def collectSbdVariables(self, ncDS):
        '''
        This function collects sbd file variables.
        '''
        #idxTime = self.data['columns']['sbd'].index('m_present_time')
        sensorTimes = self.data['sbd']['m_present_time']

        # Map sensor times to the new dimension times
        # This slow process only needs to be done once (skip if
        # we are storing this separately)
        '''
        if not(self.args.ncSeparate):
            timeDimList = list(ncDS['time'].values)
            newTimeIdx = []
            for sTime in sensorTimes:
                iSel = ncDS['time'].sel(time=sTime)
                if iSel.size == 1:
                    itime = timeDimList.index(iSel)
                else:
                    itime = timeDimList.index(iSel[0])
                newTimeIdx.append(itime)
        else:
        '''
        newTimeIdx = range(0, len(sensorTimes))
        ncDS['time'] = (("time"), sensorTimes.data)
        ncDS['time'].attrs['units'] = "seconds since 1970-01-01"
        self.fillValues = {}
        self.fillValues['time'] = {'_FillValue': -9999.9, 'dtype': 'double'}
        timeDimList = list(ncDS['time'].values)

        # Add source filename to metadata
        ncDS.attrs['source_file'] = self.data['input']['sbd']

        #self.stopToDebug()

        # Combine instrument and template variable lists
        sensorList = []
        for sensor in self.instruments:
            sensorName = sensor
            sensorList.append(sensorName)
        for sensor in self.template['variables'].keys():
            if sensor not in sensorList:
                sensorList.append(sensor)

        for sensor in self.template['variables'].keys():
            sensorName = sensor
            if sensorName in self.data['columns']['sbd']:
                #idx = self.data['columns']['sbd'].index(sensorName)
                sensorData = self.data['sbd'][sensorName]
                # Create empty array
                varName = self.obtainVariableName(sensorName)
                (fillValue, fillValueDtype) = self.obtainFillValue(varName)
                varData = np.full(len(timeDimList), fillValue)
                self.fillValues[varName] = {'_FillValue': fillValue, 'dtype': fillValueDtype}
                # Map original sensor data to larger time dimension
                scaleData = self.getScale(sensorName)
                if scaleData is not None:
                    sensorData = sensorData * scaleData
                varData[newTimeIdx] = sensorData

                # Finally, create xarray Dataset variable
                ncDS[varName] = (("time"), varData)

                # Add variable attributes
                self.addAttributes(ncDS, varName)

        return ncDS

    def collectEbdVariables(self, ncDS):
        '''
        This function collects ebd file variables.
        '''
        #idxTime = self.data['columns']['ebd'].index('sci_m_present_time')
        sensorTimes = self.data['ebd']['sci_m_present_time']

        # Map sensor times to the new dimension times
        # This slow process only needs to be done once (skip if
        # we are storing this separately)
        '''
        if not(self.args.ncSeparate):
            timeDimList = list(ncDS['time'].values)
            newTimeIdx = []
            for sTime in sensorTimes:
                iSel = ncDS['time'].sel(time=sTime)
                if iSel.size == 1:
                    itime = timeDimList.index(iSel)
                else:
                    itime = timeDimList.index(iSel[0])
                newTimeIdx.append(itime)
        else:
        '''
        newTimeIdx = range(0, len(sensorTimes))
        ncDS['time'] = (("time"), sensorTimes.data)
        ncDS['time'].attrs['units'] = "seconds since 1970-01-01"
        self.fillValues = {}
        self.fillValues['time'] = {'_FillValue': -9999.9, 'dtype': 'double'}
        timeDimList = list(ncDS['time'].values)

        # Add source filename to metadata
        ncDS.attrs['source_file'] = self.data['input']['ebd']

        #self.stopToDebug()

        # Combine instrument and template variable lists
        sensorList = []
        for sensor in self.instruments:
            sensorName = sensor
            sensorList.append(sensorName)
        for sensor in self.template['variables'].keys():
            if sensor not in sensorList:
                sensorList.append(sensor)

        for sensor in self.template['variables'].keys():
            sensorName = sensor
            if sensorName in self.data['columns']['ebd']:
                #idx = self.data['columns']['ebd'].index(sensorName)
                sensorData = self.data['ebd'][sensorName]
                # Create empty array
                varName = self.obtainVariableName(sensorName)
                (fillValue, fillValueDtype) = self.obtainFillValue(varName)
                varData = np.full(len(timeDimList), fillValue)
                self.fillValues[varName] = {'_FillValue': fillValue, 'dtype': fillValueDtype}
                # Map original sensor data to larger time dimension
                scaleData = self.getScale(sensorName)
                if scaleData is not None:
                    sensorData = sensorData * scaleData
                varData[newTimeIdx] = sensorData

                # Finally, create xarray Dataset variable
                ncDS[varName] = (("time"), varData)

                # Add variable attributes
                self.addAttributes(ncDS, varName)

        return ncDS

    def collectEchogram(self, ncDS):
        '''
        This function collects the echogram and adds it to the
        xarray Dataset object.
        '''

        sensorTimes = np.unique(self.data['spreadsheet'][:, 0])

        # Map sensor times to the new dimension times
        # This slow process only needs to be done once (skip if
        # we are storing this separately)
        if not self.args.get('ncSeparate', True):
            newTimeIdx = []
            timeDimList = list(ncDS['time'].values)
            for sTime in sensorTimes:
                iSel = ncDS['time'].sel(time=sTime)
                if iSel.size == 1:
                    itime = timeDimList.index(iSel)
                    newTimeIdx.append(itime)
                else:
                    itime = timeDimList.index(iSel[0])
                    for selTime in iSel:
                        newTimeIdx.append(itime)
                        itime = itime + 1
        else:
            sensorTimes = self.data['spreadsheet'][:, 0]
            newTimeIdx = range(0, len(sensorTimes))
            ncDS['time'] = (("time"), sensorTimes)
            ncDS['time'].attrs['units'] = "seconds since 1970-01-01"
            ncDS.attrs['source_file'] = self.data['echogram_source_file']
            self.fillValues = {}
            self.fillValues['time'] = {'_FillValue': -9999.9, 'dtype': 'double'}
            timeDimList = list(ncDS['time'].values)

        # Collect echogram_depth [:,1]
        varName = self.obtainVariableName('depth')
        (fillValue, fillValueDtype) = self.obtainFillValue(varName)
        varData = np.full(len(timeDimList), fillValue)
        self.fillValues[varName] = {'_FillValue': fillValue, 'dtype': fillValueDtype}
        # Map original sensor data to larger time dimension
        varData[newTimeIdx] = self.data['spreadsheet'][:, 1]

        # Finally, create xarray Dataset variable
        ncDS[varName] = (("time"), varData)

        # Add variable attributes
        self.addAttributes(ncDS, varName)

        # Collect echogram_sv [:,2]
        varName = self.obtainVariableName('echogram_sv')
        (fillValue, fillValueDtype) = self.obtainFillValue(varName)
        varData = np.full(len(timeDimList), fillValue)
        self.fillValues[varName] = {'_FillValue': fillValue, 'dtype': fillValueDtype}
        # Map original sensor data to larger time dimension
        varData[newTimeIdx] = self.data['spreadsheet'][:, 2]

        # Finally, create xarray Dataset variable
        ncDS[varName] = (("time"), varData)

        # Add variable attributes
        self.addAttributes(ncDS, varName)

        return ncDS

    def applyGlobalMetadata(self, ncDS):
        '''
        Generically apply global metadata data to xarray Dataset object
        '''

        # Apply deployment global metadata
        ncDS = self.applyDeploymentGlobalMetadata(ncDS)

        # Apply template global metadata
        ncDS = self.applyTemplateGlobalMetadata(ncDS)

        self.readTemplateUnlimitedDims()

        # Calculated metadata
        ncDS.attrs['date_created'] = self.dateFormat()

        return ncDS

    def writeNetCDF(self):
        '''
        Write out an IOOS DAC compatable netCDF file.
        '''

        # self.args['ncDir'] must be defined to write
        # out netcdf files
        if self.args.get('ncDir', None) is None:
            return

        # Start with an empty xarray Dataset object
        ncDS = xr.Dataset()

        timeVarTbd = None
        #timeIdxTbd = None
        if 'tbd' in self.data['columns'].keys():
            timeVarTbd = 'sci_m_present_time'
            if timeVarTbd in self.data['columns']['tbd']:
                #timeIdxTbd = self.data['columns']['tbd'].index(timeVarTbd)
                pass
            else:
                timeVarTbd = None

        timeVarSbd = None
        #timeIdxSbd = None
        if 'sbd' in self.data['columns'].keys():
            timeVarSbd = 'm_present_time'
            if timeVarSbd in self.data['columns']['sbd']:
                #timeIdxSbd = self.data['columns']['sbd'].index(timeVarSbd)
                pass
            else:
                timeVarSbd = None

        timeVarEbd = None
        #timeIdxEbd = None
        if 'ebd' in self.data['columns'].keys():
            timeVarEbd = 'sci_m_present_time'
            if timeVarEbd in self.data['columns']['ebd']:
                #timeIdxEbd = self.data['columns']['ebd'].index(timeVarEbd)
                pass
            else:
                timeVarEbd = None

        timeVarDbd = None
        #timeIdxDbd = None
        if 'dbd' in self.data['columns'].keys():
            timeVarDbd = 'm_present_time'
            if timeVarDbd in self.data['columns']['dbd']:
                #timeIdxDbd = self.data['columns']['dbd'].index(timeVarDbd)
                pass
            else:
                timeVarDbd = None

        # Only do this if we are creating one monolithic netCDF file
        # NOTE: creating one monolithic file leads to large files! Not recommended.
        '''
        if not(self.data['spreadsheet'] is None) and not(self.args.ncSeparate):
            # Try to align echogram spreadsheet with tbd or sbd times
            # This is due to reading the binary directly to decode the echogram
            # and the clipped bits produced by dbd2asc
            timeTolerance = 1e-05
            timeMap = []
            for timept in np.unique(self.data['spreadsheet'][:,0]):
                timediffTbd = np.abs(self.data['asc'][:,timeIdxTbd] - timept)
                minTimeTbd = timediffTbd.min()
                timediffSbd = np.abs(self.data['sbd'][:,timeIdxSbd] - timept)
                minTimeSbd = timediffSbd.min()
                if minTimeTbd > timeTolerance and minTimeSbd > timeTolerance:
                    sys.exit("ERROR: Time tolerance exceeded (tbd, sbd): (%f, %f)" % (minTimeTbd, minTimeSbd))

                # Use the nearest time to align the echogram Sv value
                if minTimeTbd < minTimeSbd:
                    idx = timediffTbd.argmin()
                    self.data['spreadsheet'][:,0] = np.where(self.data['spreadsheet'][:,0] == timept, \
                        self.data['asc'][idx,timeIdxTbd], self.data['spreadsheet'][:,0])
                else:
                    idx = timediffSbd.argmin()
                    self.data['spreadsheet'][:,0] = np.where(self.data['spreadsheet'][:,0] == timept, \
                        self.data['sbd'][idx,timeIdxSbd], self.data['spreadsheet'][:,0])
        '''

        # We now have to determine the full size of the time coordinate
        # with repeating times to store
        # the echogram to sci_echodroid_sv_depth
        # One row of the echogram can overlap with either a tbd or sbd data row
        timeDim = []

        #self.stopToDebug()

        # Get the shapes of 'tbd' and 'sbd'.  These data could be missing.
        timesTbd = None
        if timeVarTbd:
            shapeTbd = self.data['tbd'][timeVarTbd].shape
            if shapeTbd[0] != 0:
                timesTbd = self.data['tbd'][timeVarTbd]

        timesSbd = None
        if timeVarSbd:
            shapeSbd = self.data['sbd'][timeVarSbd].shape
            if shapeSbd[0] != 0:
                timesSbd = self.data['sbd'][timeVarSbd]

        timesEbd = None
        if timeVarEbd:
            shapeEbd = self.data['ebd'][timeVarEbd].shape
            if shapeEbd[0] != 0:
                timesEbd = self.data['ebd'][timeVarEbd]

        timesDbd = None
        if timeVarDbd:
            shapeDbd = self.data['dbd'][timeVarDbd].shape
            if shapeDbd[0] != 0:
                timesDbd = self.data['dbd'][timeVarDbd]

        #self.stopToDebug()

        # Only do this if we are creating a monolithic netCDF file
        timesPg = None
        if not self.args.get('ncSeparate', True):
            if timesTbd is not None and timesSbd is not None:
                allTimes = np.sort(np.append(timesTbd, timesSbd))
            else:
                if timesTbd is not None:
                    allTimes = timesTbd
                if timesSbd is not None:
                    allTimes = timesSbd

            if self.data['spreadsheet'] is None:
                timesPg = None
            else:
                timesPg  = self.data['spreadsheet'][:, 0]

            # Construct the time variable
            for tval in allTimes:
                # If a time overlaps with the echogram, use the block of times from
                # the echogram otherwise keep a single value
                #self.stopToDebug()
                if timesPg is None:
                    nPg = 0
                else:
                    nPg = len(np.where(timesPg == tval)[0])
                if nPg == 0:
                    timeDim.append(tval)
                else:
                    [timeDim.append(tval) for i in range(0, nPg)]

            # Now we collect individual variables with the given time dimension
            ncDS['time'] = (("time"), timeDim)
            self.fillValues['time'] = {'_FillValue': -9999.9, 'dtype': 'double'}
        else:
            if self.data['spreadsheet'] is not None:
                timesPg  = self.data['spreadsheet'][:, 0]

        # Collect tbd variables
        if timesTbd is not None:
            if self.debugFlag:
                print("Collecting tbd variables")
            ncDS = self.collectTbdVariables(ncDS)
            if self.args.get('ncSeparate', True):
                #timeShape = ncDS['time'].shape
                timeStamp = self.dateFormat(
                    datetime.datetime.utcfromtimestamp(ncDS['time'].min().values + 0.0),
                    fmt="%Y%m%d_%H%M%S")
                tbdFilename = os.path.join(self.args['ncDir'], "%s_tbd.nc" % (timeStamp))
                # Backfill variables to maintain consistency in the file
                timeDimList = list(ncDS['time'].values)
                ensureVariables = [
                    'temperature', 'sci_flbbcd_bb_units', 'sci_flbbcd_chlor_units', 'sci_rinkoii_do',
                    'sci_water_cond', 'sci_flbbcd_cdom_units', 'sci_water_temp',
                    'pressure', 'sci_water_pressure', 'sci_echodroid_sv', 'sci_echodroid_sa', 'sci_echodroid_propocc',
                    'sci_echodroid_aggindex', 'sci_echodroid_ctrmass', 'sci_echodroid_inertia', 'sci_echodroid_eqarea']
                for varName in ensureVariables:
                    if varName not in ncDS.variables:
                        (fillValue, fillValueDtype) = self.obtainFillValue(varName)
                        varData = np.full(len(timeDimList), fillValue)
                        self.fillValues[varName] = {'_FillValue': fillValue, 'dtype': fillValueDtype}
                        ncDS[varName] = (("time"), varData)
                ncDS = self.applyGlobalMetadata(ncDS)
                if self.debugFlag:
                    print("Writing netCDF: %s" % (tbdFilename))
                ncDS.to_netcdf(tbdFilename, unlimited_dims=self.ncUnlimitedDims, encoding=self.fillValues)
                # Reset after saving tbd netCDF
                ncDS = xr.Dataset()

        # Collect dbd variables
        if timesDbd is not None:
            if self.debugFlag:
                print("Collecting dbd variables")
            ncDS = self.collectDbdVariables(ncDS)
            if self.args.get('ncSeparate', True):
                #timeShape = ncDS['time'].shape
                timeStamp = self.dateFormat(
                    datetime.datetime.utcfromtimestamp(ncDS['time'].min().values + 0.0),
                    fmt="%Y%m%d_%H%M%S")
                dbdFilename = os.path.join(self.args['ncDir'], "%s_dbd.nc" % (timeStamp))
                # Backfill variables to maintain consistency in the file
                timeDimList = list(ncDS['time'].values)
                ensureVariables = [
                    'm_heading', 'm_lat', 'm_lon', 'm_pitch', 'm_depth', 'c_wpt_lon', 'c_wpt_lat',
                    'm_roll', 'm_battery', 'm_columb_amphr_total', 'm_gps_lat', 'm_gps_lon', 'm_coulomb_amphr_total',
                    'm_ballast_pumped']
                for varName in ensureVariables:
                    if varName not in ncDS.variables:
                        (fillValue, fillValueDtype) = self.obtainFillValue(varName)
                        varData = np.full(len(timeDimList), fillValue)
                        self.fillValues[varName] = {'_FillValue': fillValue, 'dtype': fillValueDtype}
                        ncDS[varName] = (("time"), varData)
                ncDS = self.applyGlobalMetadata(ncDS)
                if self.debugFlag:
                    print("Writing netCDF: %s" % (dbdFilename))
                ncDS.to_netcdf(dbdFilename, unlimited_dims=self.ncUnlimitedDims, encoding=self.fillValues)
                # Reset after saving dbd netCDF
                ncDS = xr.Dataset()

        # Collect sbd variables
        if timesSbd is not None:
            if self.debugFlag:
                print("Collecting sbd variables")
            ncDS = self.collectSbdVariables(ncDS)
            if self.args.get('ncSeparate', True):
                #timeShape = ncDS['time'].shape
                timeStamp = self.dateFormat(
                    datetime.datetime.utcfromtimestamp(ncDS['time'].min().values + 0.0),
                    fmt="%Y%m%d_%H%M%S")
                sbdFilename = os.path.join(self.args['ncDir'], "%s_sbd.nc" % (timeStamp))
                # Backfill variables to maintain consistency in the file
                timeDimList = list(ncDS['time'].values)
                ensureVariables = [
                    'm_heading', 'm_lat', 'm_lon', 'm_pitch', 'm_depth', 'c_wpt_lon', 'c_wpt_lat',
                    'm_roll', 'm_battery', 'm_columb_amphr_total', 'm_gps_lat', 'm_gps_lon', 'm_coulomb_amphr_total',
                    'm_ballast_pumped']
                for varName in ensureVariables:
                    if varName not in ncDS.variables:
                        (fillValue, fillValueDtype) = self.obtainFillValue(varName)
                        varData = np.full(len(timeDimList), fillValue)
                        self.fillValues[varName] = {'_FillValue': fillValue, 'dtype': fillValueDtype}
                        ncDS[varName] = (("time"), varData)
                ncDS = self.applyGlobalMetadata(ncDS)
                if self.debugFlag:
                    print("Writing netCDF: %s" % (sbdFilename))
                ncDS.to_netcdf(sbdFilename, unlimited_dims=self.ncUnlimitedDims, encoding=self.fillValues)
                # Reset after saving tbd netCDF
                ncDS = xr.Dataset()

        # Collect ebd variables
        if timesEbd is not None:
            if self.debugFlag:
                print("Collecting ebd variables")
            ncDS = self.collectEbdVariables(ncDS)
            if self.args.get('ncSeparate', True):
                #timeShape = ncDS['time'].shape
                timeStamp = self.dateFormat(
                    datetime.datetime.utcfromtimestamp(ncDS['time'].min().values + 0.0),
                    fmt="%Y%m%d_%H%M%S")
                ebdFilename = os.path.join(self.args['ncDir'], "%s_ebd.nc" % (timeStamp))
                # Backfill variables to maintain consistency in the file
                timeDimList = list(ncDS['time'].values)
                ensureVariables = [
                    'temperature', 'sci_flbbcd_bb_units', 'sci_flbbcd_chlor_units', 'sci_rinkoii_do',
                    'sci_water_cond', 'sci_flbbcd_cdom_units', 'sci_water_temp',
                    'pressure', 'sci_water_pressure', 'sci_echodroid_sv', 'sci_echodroid_sa', 'sci_echodroid_propocc',
                    'sci_echodroid_aggindex', 'sci_echodroid_ctrmass', 'sci_echodroid_inertia', 'sci_echodroid_eqarea']
                for varName in ensureVariables:
                    if varName not in ncDS.variables:
                        (fillValue, fillValueDtype) = self.obtainFillValue(varName)
                        varData = np.full(len(timeDimList), fillValue)
                        self.fillValues[varName] = {'_FillValue': fillValue, 'dtype': fillValueDtype}
                        ncDS[varName] = (("time"), varData)
                ncDS = self.applyGlobalMetadata(ncDS)
                if self.debugFlag:
                    print("Writing netCDF: %s" % (ebdFilename))
                ncDS.to_netcdf(ebdFilename, unlimited_dims=self.ncUnlimitedDims, encoding=self.fillValues)
                # Reset after saving tbd netCDF
                ncDS = xr.Dataset()

        # Collect echogram
        if timesPg is not None:
            if self.debugFlag:
                print("Collecting echogram")
            ncDS = self.collectEchogram(ncDS)
            if self.args.get('ncSeparate', True):
                #timeShape = ncDS['time'].shape
                timeStamp = self.dateFormat(
                    datetime.datetime.utcfromtimestamp(ncDS['time'].min().values + 0.0),
                    fmt="%Y%m%d_%H%M%S")
                svFilename = os.path.join(self.args['ncDir'], "%s_sv.nc" % (timeStamp))
                ncDS = self.applyGlobalMetadata(ncDS)

                # One last adjustment to filename if reprocessing is being used
                if self.data['process']:
                    svFilename = self.appendFilenameSuffix(svFilename, self.data['process'])

                if self.debugFlag:
                    print("Writing netCDF: %s" % (svFilename))
                ncDS.to_netcdf(svFilename, unlimited_dims=self.ncUnlimitedDims, encoding=self.fillValues)
                # Reset after saving tbd netCDF
                ncDS = xr.Dataset()

        if not self.args.get('ncSeparate', True):
            ncDS = self.applyGlobalMetadata(ncDS)
            timeStamp = self.dateFormat(
                datetime.datetime.utcfromtimestamp(ncDS['time'].min().values + 0.0),
                fmt="%Y%m%d_%H%M%S")
            if self.debugFlag:
                print("Writing netCDF: %s" % (self.ncDir))
            ncDS.to_netcdf(
                os.path.join(self.ncDir, "%s.nc" % (timeStamp)),
                unlimited_dims=self.ncUnlimitedDims, encoding=self.fillValues)
