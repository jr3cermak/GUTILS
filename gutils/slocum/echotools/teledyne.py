import io, os, sys, struct, datetime
import subprocess
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Colormap
import matplotlib.dates as dates
import matplotlib.ticker as mticker
from matplotlib.patches import Rectangle

class Glider:
    '''
    A container class for handling Teledyne Webb glider data.
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
        self.tbdFile = tbdFile
        self.sbdFile = sbdFile
        self.cacheDir = cacheDir
        self.dbd2asc = dbd2asc
        self.debugFlag = debugFlag

        self.data = {
            'asc': None,
            'sbd': None,
            'pseudogram': None,
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

    # Generic functions

    def hatches_plot(self, ax, h):
        '''
        Create a default background for glider plots 'o'.
        '''
        ax.add_patch(Rectangle((0, 0), 2000, 2000, fill=False, hatch=h, alpha=0.25, zorder=1))

        return

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
        array = np.asarray(array)
        idx = (np.abs(array - val)).argmin()

        return idx

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
            except:
                import pdb; pdb.set_trace()

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
                if source == 'asc':
                    ind = self.data['columns'].index(col)
                if source == 'sbd':
                    ind = self.data['sbdcolumns'].index(col)
            except:
                pass

            if ind != -1:
                data = self.data[source][:, ind]
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
            except:
                pass

            if ind != -1:
                selectedData = selectedData[~np.isnan(selectedData[:,ind]),:]

        # Split columns into a dictionary if requested
        if asDict:
            colCount = 0
            dictData = {}
            for col in columns:
                dictData[col] = selectedData[:,colCount]
                colCount = colCount + 1
            selectedData = dictData

        return selectedData

    def handleSpreadsheet(self, args):
        '''
        This function handles writing out the decoded tbd data
        in CSV format.  Either a filename is provided or the
        spreadsheet is sent to 'stdout'.  If the csvHeader flag
        is set to True, a header is also provided.

        Parameters
        ----------
        args : :obj:`argparse`
            Parsed command line arguments.

        Notes
        -----
            * args.debugFlag: Boolean flag.  If True, additional output is printed
              to standard output.
            * args.csvOut: May be a full or relative path with filename or `stdout`.
            * args.csvHeader: Boolean flag.  If True, a header is included with
              CSV output to file or standard out.
        '''

        # DEBUG
        if self.debugFlag:
            print("DEBUG: args.csvOut:", args.csvOut)
            print("DEBUG: args.csvHeader:", args.csvHeader)

        # If this argument is None, do not run this function
        if args.csvOut is None:
            return

        # If args.csvOut is not "stdout" assume this is a filename
        outptr = sys.stdout
        stdoutFlag = True

        # Skip output if file fails to open for writing
        if args.csvOut != "stdout":

            # Use the input filename as the csv output filename
            try:
                outptr = open(args.csvOut, "w")
                stdoutFlag = False
            except:
                print("WARNING: Unable to open CSV output file for writing, skipping.")
                return

        if args.csvHeader:
            outptr.write("Timestamp(seconds since 1970-1-1), Depth(meters), Sv(dB)\n")

        for data in self.data['spreadsheet']:
            outptr.write("%f, %f, %f\n" % (data[0], data[1], data[2]))

        if not(stdoutFlag):
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

    def handleImage(self, args):
        '''
        This function handles writing out a graphical image.  By
        default, the image rendering uses descrete pixels that
        have depth and time bins.  If --useScatterPlot is set,
        a scatter plot is produced instead of a time/depth binned
        (raster image/imshow) plot.  The raster plot coordinates
        are the depth and time bins which requires redefining x
        and y labels on the fly.

        Parameters
        ----------
        args : :obj:`argparse`
            Parsed command line arguments.

        Notes
        -----
            * args.imageOut: May be a full or relative path with filename or `stdout`.
            * args.debugFlag: Boolean flag.  If True, additional output is printed
              to standard output.
            * args.useScatterPlot: Boolean flag.  If True, a scatter plot is
              produced instead of the depth/time binned plot.
        '''

        # If args.imageOut is not "stdout" assume this is a filename
        stdoutFlag = True

        if args.imageOut != "stdout":
            stdoutFlag = False

        # Set the default SIMRAD EK500 color table plus grey for NoData.
        simrad_color_table = [(1, 1, 1),
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
            (0.4705, 0.2353, 0.1568)]
        simrad_cmap = (LinearSegmentedColormap.from_list
            ('Simrad', simrad_color_table))
        simrad_cmap.set_bad(color='lightgrey')

        depthBinSize = int(abs(self.data['depthBinLength']))
        timeBinSize = self.data['timeBinLength']

        # Spreadsheet columns
        # [0] Timestamp(seconds since 1970-1-1, [1] depth(meters), [2] Sv (dB)
        dataSpreadsheet = self.data['spreadsheet']

        if dataSpreadsheet is None:
            print("WARNING: No usable data found!")
            if self.sbdFile is None:
                print("HINT: Include a sbd file if available.")
            return

        # Calculate how many depth pixels (bins) we need given spreadsheet data
        # minDepth = shallowest pixel
        # maxDepth = deepest pixel
        minDepth = int(dataSpreadsheet[:,1].min() / depthBinSize)
        maxDepth = int(np.ceil(dataSpreadsheet[:,1].max() / depthBinSize))

        if self.debugFlag:
            print("Depth bins: Min(%d) Max(%d)" % (minDepth, maxDepth))

        numberOfDepthPixels = (maxDepth-minDepth)+1
        depthTicks = []
        for depthPixel in range(0, numberOfDepthPixels, 5):
            startDepth = minDepth*depthBinSize + depthPixel*depthBinSize
            endDepth = startDepth + depthBinSize
            midPixelDepth = (startDepth + endDepth) / 2.0
            depthTicks.append(midPixelDepth)

        # If the user selects a scatter plot, do that now and ignore the rest of the code
        if args.useScatterPlot:

            # Sort Sv(dB) from lowest to highest so higher values are plotted last
            dataSpreadsheet = dataSpreadsheet[np.argsort(dataSpreadsheet[:,2])]

            # Plot simply x, y, z data (time, depth, dB)
            fig, ax = plt.subplots(figsize=(10,8))
            ax.xaxis.set_minor_locator(dates.MinuteLocator(interval=10))   # every 15 minutes
            ax.xaxis.set_minor_formatter(dates.DateFormatter('%H:%M'))  # hours and minutes
            ax.xaxis.set_major_locator(dates.DayLocator(interval=1))    # every day
            ax.xaxis.set_major_formatter(dates.DateFormatter('\n%m-%d-%Y'))
            #ax.set_facecolor((0.4705, 0.2353, 0.1568))
            ax.set_facecolor('lightgray')
            #self.hatches_plot(ax, 'o')
            im = plt.scatter(dates.epoch2num(dataSpreadsheet[:,0]), dataSpreadsheet[:,1], c=dataSpreadsheet[:,2],
                    cmap=simrad_cmap, s=20.0, zorder=2)
            cbar = plt.colorbar(orientation='vertical', label='Sv (dB)', shrink=0.35)
            plt.gca().invert_yaxis()
            plt.ylabel('depth (m)')
            plt.xlabel('Date/Time (UTC)')
            plt.clim(0, -60)
            if args.title:
                plt.title(args.title)
            # Determine if we are writing to stdout
            if stdoutFlag:
                plt.savefig(sys.stdout.buffer, bbox_inches='tight', dpi=100)
            else:
                # Plot image
                plt.savefig(args.imageOut, bbox_inches='tight', dpi=100)

            return

        # We need to know the time indexes for the final graphic below too
        timeIndexes = np.unique(dataSpreadsheet[:,0])

        # Loop through each time on the spreadsheet
        numDataRecords = 0
        for timeIdx in timeIndexes:
            sample = np.ones((numberOfDepthPixels,)) * -90.0

            cond = np.where(dataSpreadsheet[:,0]==timeIdx)[0]
            for recIdx in cond:
                (timeRec, depthRec, dBRec) = dataSpreadsheet[recIdx,:]
                depthBin = self.getDepthPixel(depthRec, minDepth, maxDepth, depthBinSize)
                if self.debugFlag:
                    print("  Timestamp(%.1f) Depth(%.1f meters) Sv(%.1f dB) DepthPixel(%d)" %\
                            (timeRec, depthRec, dBRec, depthBin))
                sample[depthBin] = dBRec
            if numDataRecords == 0:
                imageData = sample
            else:
                imageData = np.vstack((imageData, sample))
            numDataRecords = numDataRecords + 1

        # Create image plot
        # Axis labels are centered on the pixel

        fig, ax = plt.subplots(figsize=(10,8))
        #self.hatches_plot(ax, 'X')
        #imageData = np.ma.masked_where(imageData < -90.0, imageData)
        imageData[np.where(imageData == -90.0)] = np.nan
        plotData = np.transpose(imageData)
        #im = plt.imshow(plotData, cmap=simrad_cmap, zorder=2)
        im = plt.imshow(plotData, cmap=simrad_cmap, interpolation='none')
        cbar = plt.colorbar(orientation='vertical', label='Sv (dB)', shrink=0.35)

        # x and y axis labels
        plt.ylabel('depth (m)')
        plt.xlabel('Date/Time (UTC)')
        plt.clim(0, -60)

        # Adjust x tick labels: time bin -> time string
        # Make sure these ticks line up with integer time bins
        xtickLabels = []
        xtickLocs = ax.get_xticks()

        xtickLocsNew = []
        for timeTick in xtickLocs:
            # Skip non integer time ticks
            if int(timeTick) != timeTick:
                continue

            if int(timeTick) >= 0.0 and int(timeTick) < len(timeIndexes):
                startTime = timeIndexes[int(timeTick)]
                endTime = startTime + timeBinSize
                midPixelTime = (startTime + endTime) / 2.0
                xtickNew = datetime.datetime.utcfromtimestamp(midPixelTime).strftime("%Y-%m-%d\n%H:%M:%S")
            else:
                xtickNew = "%s" % (timeTick)

            xtickLocsNew.append(timeTick)
            xtickLabels.append(xtickNew)

        ax.xaxis.set_major_locator(mticker.FixedLocator(xtickLocsNew))
        ax.set_xticklabels(xtickLabels)
        plt.xticks(rotation = 45.0)

        # Adjust y tick labels: depth bin -> depth (meters)
        ytickLabels = []
        ytickLocs = ax.get_yticks()
        for depthTick in ytickLocs:
            startDepth = minDepth*depthBinSize + (depthTick*depthBinSize)
            endDepth = startDepth + depthBinSize
            midPixelDepth = (startDepth + endDepth) / 2.0
            ytickNew = ("%.1f" % (midPixelDepth))
            ytickLabels.append(ytickNew)

        if not(args.binnedDepthLabels):
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
                ytickLabels.append(str(depth))

            #self.stopToDebug()

        ax.yaxis.set_major_locator(mticker.FixedLocator(ytickLocs))
        ax.set_yticklabels(ytickLabels)

        # Set plot title (if available)
        if args.title:
            plt.title(args.title)

        # Determine if we are writing to stdout
        if stdoutFlag:
            plt.savefig(sys.stdout.buffer, bbox_inches='tight', dpi=100)
        else:
            # Save plot image to specified filename
            plt.savefig(args.imageOut, bbox_inches='tight', dpi=100)

    # Glider functions

    def createPseudogramSpreadsheet(self, args):
        '''
        This function reads GLIDER.data['pseudogram'] and
        places it in a spreadsheet format in
        GLIDER.data['spreadsheet'].  The function is
        expecting at least two fields to have been read
        from the provided tbd file.

        Parameters
        ----------
        args : :obj:`argparse`
            Parsed command line arguments.

        Notes
        -----
            * args.debugFlag: Boolean flag.  If True, additional
              output is printed to standard output.
            * args.useScatterPlot: Boolean flag.  If True, a
              scatter plot is produced instead of the depth/time
              binned plot.
        '''

        # Code from echoGenNew.py
        eData = self.data['pseudogram']

        # We extract the columns from decoded dbd2asc instead of
        # requiring another Teledyne program (dba_sensor_filter)
        barData = self.extractColumns('asc', columns=['sci_m_present_time','sci_water_pressure'],
            ignoreNaNColumns=['sci_water_pressure'])

        # Change bars to meters
        barData[:, 1] = barData[:, 1] * 10.0

        # If a sbd data file is given, merge the sbd data with the
        # barData.
        if self.sbdFile:
           depthData = self.extractColumns('sbd', columns=['m_present_time', 'm_depth'],
               ignoreNaNColumns=['m_depth'])
           barData = np.append(barData, depthData, axis=0)
           barData = barData[np.argsort(barData[:,0])]

        # Separate time from ping (echosound) data
        pingTime = eData[:,0]
        # Columns 1 -> 21
        #   Assuming values are increasing with depth
        pingData = eData[:,1:]

        # Separate time from depth (bar) data
        depthTimes = barData[:,0]
        # Separate depth from depth (bar) data
        depths = barData[:,1]

        # Echosounder range (meters)
        # Positive values: instrument is pointed down
        # Negative values: instrument is pointed up
        echosounderRange = float(args.echosounderRange)
        # Number of depth bins (0-19)
        numberDepthBins = 20
        depthBins = range(0,numberDepthBins)
        # Currently fixed at 3.0 meters
        depthBinLength = echosounderRange / numberDepthBins

        # Determine time range to plot based on depthTimes
        if len(depthTimes) == 0:
            print("WARNING: No usable depth information found for %s" % (args.tbdFile))
            sys.exit()

        # Average time between depths (should be about 8 seconds)
        # If we assume 8.0 seconds, then the time bin is
        # timeBin - 4.0 seconds to timeBin + 4.0 seconds

        # This is the size of the time bin in which pings can be
        # collected.
        avgDepthReading = ((depthTimes[-1]-depthTimes[0])/len(depthTimes)) / 2.0
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
            print("  Instrument range: %f meters" % (echosounderRange))
            print("  Depth bin size: %f meters" % (depthBinLength))

        # Run through each of the scans and prepare a data table/spreadsheet

        # If a time index on a set of pings is the same, increment the
        # plotting time index by 10 seconds to obtain proper alignment.
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
            if abs(timeOffset) > (avgDepthReading*2.0):
                if self.debugFlag:
                    print("  WARNING: Ping (%d) skipped: Ping time (%f) Offset (%f seconds) > Time Bin (%f seconds)" %\
                        (pingNo, pingTime[pingNo], timeOffset, avgDepthReading))
                continue

            # If this ping is the same time, then we need to apply a 10 second offset to
            # the new set of ping data.  This is for the "echo" mode.
            if lastPingTime == pingTime[pingNo]:
                pingOffset = pingOffset + 10.0
            else:
                pingOffset = 0.0
                lastPingTime = pingTime[pingNo]

            if self.debugFlag:
                print("  Ping (%d) Ping time (%f) Depth time (%f) Offset (%f seconds) Depth (%f meters)" % \
                    (pingNo, pingTime[pingNo]+pingOffset, depthTime, timeOffset, selectedDepth))

            # Process ping data is always closest to farthest (0, maxRange)
            # maxRange (-) if instrument is pointed upwards (shallower)
            # maxRange (+) if instrument is pointed downwards (deeper)
            pingDepth = selectedDepth - depthBinLength
            for ping in pingData[pingNo]:
                pingDepth = pingDepth + depthBinLength

                # If pingDepth is above the surface, ignore it
                if pingDepth < 0.0:
                    continue

                if self.debugFlag:
                    print("      Sv (%f dB) depth (%f meters)" % (ping, pingDepth))

                dataRow = np.array((pingTime[pingNo]+pingOffset, pingDepth, ping))
                if dataCount == 0:
                    dataSpreadsheet = np.column_stack(dataRow)
                else:
                    dataSpreadsheet = np.append(dataSpreadsheet, np.column_stack(dataRow), axis=0)
                dataCount = dataCount + 1

        self.data['depthBinLength'] = depthBinLength
        self.data['timeBinLength'] = timeBinLength
        self.data['spreadsheet'] = dataSpreadsheet

        return

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
        for i in range(0,14):
            line = tbdfp.readline()
            lnParts = line.decode('utf-8','replace').split(':')
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

        cacheFile = os.path.join(self.cacheDir, '%s.cac' %\
            (self.data['cacheMetadata']['cacheFile']))

        if not(os.path.isfile(cacheFile)):
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
            if not(ln):
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
        for i in range(0,14):
            line = sbdfp.readline()
            lnParts = line.decode('utf-8','replace').split(':')
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

        cacheFile = os.path.join(self.cacheDir, '%s.cac' %\
            (self.data['sbdMetadata']['cacheFile']))

        if not(os.path.isfile(cacheFile)):
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
            if not(ln):
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

    def readPseudogram(self):
        '''
        This function reads the glider tbd file and extracts the embedded
        pseudogram from the echometrics data.  The `dbd2asc` file cannot be used
        since it truncates the least significant bits in which the pseudogram is
        embedded.

        NOTE: This function `readPsuedogram` should only be used for
        extracting encoded pseudogram information embedded in the echometrics
        data.  All other glider files should be read using `dbd2asc`.

        This function automatically determines if the glider
        was in "echo" or "combo" mode.  Prior knowledge of the
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
                print("Glider.readPseudogram(): Binary read error.")
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
                    if numRecords == 0:
                        self.data['pseudogram'] = np.column_stack(dataRec)
                    else:
                        self.data['pseudogram'] = np.append(self.data['pseudogram'], np.column_stack(dataRec), axis=0)
                    numRecords = numRecords + 1

        bdFile.close()

        # Perform reorder step
        (numRows, numCols) = self.data['pseudogram'].shape

        # Density (Sv) (dB) bins are categorized 0 to 7
        bins = {0: '-5.0', 1:'-15.0', 2:'-22.5', 3:'-27.5', 4:'-32.5', 5:'-40.0', 6:'-50.0', 7:'-60.0' }

        # Data is hidden in the last three bits of data sent over iridium
        #mask of 7(binary 111) for 3bit vals
        mask = 7

        # reord.py reads off the first row to skip it
        for row in range(1, numRows):
            words = self.data['pseudogram'][row, :]
            if len(words) != 0:
                #print(words)
                if int(words[2]) == -1032813281:
                    # This decodes regular echometrics mode
                    #print('echo')
                    #time
                    #print(words[7] + ' ', end=''),
                    #print("%f " % (words[7]), end=''),
                    decodedPseudogram = np.array(words[7])
                    #decodedPseudogram = np.array(words[7])
                    #2 32bit vals two high bits are don't cares, read off 3 at time
                    for i in range(0, 30, 3):
                        #print(str((int(words[6])>>i) & mask) + ' ', end=''),
                        #print(bins[(int(words[6])>>i) & mask] + ' ', end=''),
                        #print("%s " % (bins[(int(words[6])>>i) & mask]), end=''),
                        decodedPseudogram = np.append(decodedPseudogram, (bins[(int(words[6])>>i) & mask]))
                    for i in range(0, 30, 3):
                        #print(str((int(words[3])>>i) & mask) + ' ', end=''),
                        #print(bins[(int(words[3])>>i) & mask] + ' ', end=''),
                        #print("%s " % (bins[(int(words[3])>>i) & mask]), end=''),
                        decodedPseudogram = np.append(decodedPseudogram, (bins[(int(words[3])>>i) & mask]))
                    #print('\n')
                    if row == 1:
                        fullPseudogram = np.column_stack(decodedPseudogram)
                    else:
                        fullPseudogram = np.append(fullPseudogram, np.column_stack(decodedPseudogram), axis=0)
                    #time
                    #print(words[7] + ' ', end=''),
                    #print("%f " % (words[7]), end=''),
                    decodedPseudogram = np.array(words[7])
                    #2 32bit vals two high bits are don't cares, read off 3 at time
                    for i in range(0, 30, 3):
                        #print(str((int(words[0])>>i) & mask) + ' ', end=''),
                        #print(bins[(int(words[0])>>i) & mask] + ' ', end=''),
                        #print("%s " % (bins[(int(words[0])>>i) & mask]), end=''),
                        decodedPseudogram = np.append(decodedPseudogram, (bins[(int(words[0])>>i) & mask]))
                    for i in range(0, 30, 3):
                        #print(str((int(words[5])>>i) & mask) + ' ', end=''),
                        #print(bins[(int(words[5])>>i) & mask] + ' ', end=''),
                        #print("%s " % (bins[(int(words[5])>>i) & mask]), end=''),
                        decodedPseudogram = np.append(decodedPseudogram, (bins[(int(words[5])>>i) & mask]))
                    #print('\n')
                    if row == 1:
                        fullPseudogram = np.append(fullPseudogram, np.column_stack(decodedPseudogram), axis=0)
                    else:
                        fullPseudogram = np.append(fullPseudogram, np.column_stack(decodedPseudogram), axis=0)
                    #time
                    #print(words[7] + ' ', end=''),
                    #print("%f " % (words[7]), end=''),
                    decodedPseudogram = np.array(words[7])
                    #2 32bit vals two high bits are don't cares, read off 3 at time
                    for i in range(0, 30, 3):
                        #print(str((int(words[1])>>i) & mask) + ' ', end=''),
                        #print(bins[(int(words[1])>>i) & mask] + ' ', end=''),
                        #print("%s " % (bins[(int(words[1])>>i) & mask]), end=''),
                        decodedPseudogram = np.append(decodedPseudogram, (bins[(int(words[1])>>i) & mask]))
                    for i in range(0, 30, 3):
                        #print(str((int(words[4])>>i) & mask) + ' ', end=''),
                        #print(bins[(int(words[4])>>i) & mask] + ' ', end=''),
                        #print("%s " % (bins[(int(words[4])>>i) & mask]), end=''),
                        decodedPseudogram = np.append(decodedPseudogram, (bins[(int(words[4])>>i) & mask]))
                    #print('\n')
                    if row == 1:
                        fullPseudogram = np.append(fullPseudogram, np.column_stack(decodedPseudogram), axis=0)
                    else:
                        fullPseudogram = np.append(fullPseudogram, np.column_stack(decodedPseudogram), axis=0)
                else:
                    # This decodes combo mode
                    #print('combo')
                    #time
                    #print("%f " % (words[7]), end=''),
                    decodedPseudogram = np.array(words[7])
                    #print(words[7] + ' ', end=''),
                    #2 32bit vals are metrics with LS 9bits used for 3 layer values
                    for i in range(0, 9, 3):
                        #print(str((int(words[6])>>i) & mask) + ' ', end=''),
                        #print(bins[(int(words[6])>>i) & mask] + ' ', end=''),
                        #print("%s " % (bins[(int(words[6])>>i) & mask]), end=''),
                        decodedPseudogram = np.append(decodedPseudogram, (bins[(int(words[6])>>i) & mask]))
                    for i in range(0, 9, 3):
                        #print(str((int(words[4])>>i) & mask) + ' ', end=''),
                        #print(bins[(int(words[4])>>i) & mask] + ' ', end=''),
                        #print("%s " % (bins[(int(words[4])>>i) & mask]), end=''),
                        decodedPseudogram = np.append(decodedPseudogram, (bins[(int(words[4])>>i) & mask]))
                    for i in range(0, 9, 3):
                        #print(str((int(words[0])>>i) & mask) + ' ', end=''),
                        #print(bins[(int(words[0])>>i) & mask] + ' ', end=''),
                        #print("%s " % (bins[(int(words[0])>>i) & mask]), end=''),
                        decodedPseudogram = np.append(decodedPseudogram, (bins[(int(words[0])>>i) & mask]))
                    for i in range(0, 9, 3):
                        #print(str((int(words[5])>>i) & mask) + ' ', end=''),
                        #print(bins[(int(words[5])>>i) & mask] + ' ', end=''),
                        #print("%s " % (bins[(int(words[5])>>i) & mask]), end=''),
                        decodedPseudogram = np.append(decodedPseudogram, (bins[(int(words[5])>>i) & mask]))
                    for i in range(0, 9, 3):
                        #print(str((int(words[1])>>i) & mask) + ' ', end=''),
                        #print(bins[(int(words[1])>>i) & mask] + ' ', end=''),
                        #print("%s " % (bins[(int(words[1])>>i) & mask]), end=''),
                        decodedPseudogram = np.append(decodedPseudogram, (bins[(int(words[1])>>i) & mask]))
                    for i in range(0, 9, 3):
                        #print(str((int(words[3])>>i) & mask) + ' ', end=''),
                        #print(bins[(int(words[3])>>i) & mask] + ' ', end=''),
                        #print("%s " % (bins[(int(words[3])>>i) & mask]), end=''),
                        decodedPseudogram = np.append(decodedPseudogram, (bins[(int(words[3])>>i) & mask]))
                    for i in range(0, 9, 3):
                        #print(str((int(words[2])>>i) & mask) + ' ', end=''),
                        #print(bins[(int(words[2])>>i) & mask] + ' ', end=''),
                        #print("%s " % (bins[(int(words[2])>>i) & mask]), end=''),
                        decodedPseudogram = np.append(decodedPseudogram, (bins[(int(words[2])>>i) & mask]))
                    #print('\n')
                    if row == 1:
                        fullPseudogram = np.column_stack(decodedPseudogram)
                    else:
                        fullPseudogram = np.append(fullPseudogram, np.column_stack(decodedPseudogram), axis=0)

        # Return the final pseudogram data
        fullPseudogram = fullPseudogram.astype(float)
        self.data['pseudogram'] = fullPseudogram
