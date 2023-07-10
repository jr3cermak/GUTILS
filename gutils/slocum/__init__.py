#!/usr/bin/env python
import datetime
import dbdreader
import os
import pyarrow
import sys
import shutil
import warnings
from glob import glob
from pathlib import Path
from tempfile import mkdtemp
from collections import OrderedDict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from gsw import z_from_p, p_from_z

from gutils import (
    generate_stream,
    get_decimal_degrees,
    interpolate_gps,
    masked_epoch,
    read_attrs,
    safe_makedirs
)
from gutils.ctd import calculate_practical_salinity, calculate_density

from pocean.utils import (
    dict_update
)

import logging
logging.getLogger("dbdreader").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
L = logging.getLogger(__name__)


"""
MODE_MAPPING = {
    "rt": ["sbd", "tbd", "mbd", "nbd"],
    "delayed": ["dbd", "ebd"]
}
"""
ALL_EXTENSIONS = [".sbd", ".tbd", ".mbd", ".nbd", ".dbd", ".ebd"]


COMPUTE_PRESSURE = 1
USE_RAW_PRESSURE = 2


class SlocumReader(object):

    TIMESTAMP_SENSORS = ['m_present_time', 'sci_m_present_time']
    PRESSURE_SENSORS = ['sci_water_pressure', 'sci_water_pressure2', 'm_water_pressure', 'm_pressure']
    DEPTH_SENSORS = ['m_depth', 'm_water_depth']
    TEMPERATURE_SENSORS = ['sci_water_temp', 'sci_water_temp2']
    CONDUCTIVITY_SENSORS = ['sci_water_cond', 'sci_water_cond2']

    def __init__(self, ascii_file, kwargs=dict()):
        self.ascii_file = ascii_file
        self.kwargs = kwargs

        # Set the PARQUET flag
        use_parquet = self.kwargs.get('enable_parquet', False)
        self.is_pq = use_parquet
        if use_parquet:
            L.debug("PARQUET enabled deployment (SlocumReader)")

        self.metadata, self.data = self.read()

        # Extra processing (echograms and other extra items)
        self._extras = pd.DataFrame()

        # Set the mode to 'rt' or 'delayed'
        # Use the directory names passed with "file".
        self.mode = ascii_file.split("/")[-3]
        """
        self.mode = self.kwargs.get('process_mode', None)
        if 'filename_extension' in self.metadata:
            filemode = self.metadata['filename_extension']
            for m, extensions in MODE_MAPPING.items():
                if filemode in extensions:
                    self.mode = m
                    break
        """

    def extras(self, data):
        """
        Extra data processing for auxilary or experimental data.  This
        section is driven by arguments to kwargs supplied by
        deployment.json(extra_kwargs).  NOTE: kwargs now passed at
        class creation.  See: self.kwargs

        Available settings:
          * echometrics with echograms ("echograms": "enable_nc": true)
            This processes echogram and echometrics variables stores them
            using an extras time dimension.
        """

        ECHOMETRICS_SENSORS = [
            'sci_echodroid_aggindex',
            'sci_echodroid_ctrmass',
            'sci_echodroid_eqarea',
            'sci_echodroid_inertia',
            'sci_echodroid_propocc',
            'sci_echodroid_sa',
            'sci_echodroid_sv',
        ]

        ECHOGRAM_VARS = [
            'echogram_sv',
        ]

        ECHOGRAM_CSV_COLUMNS = [
            'echogram_time',
            'echogram_depth',
            'echogram_sv',
        ]

        # Default extra settings
        kwargs = self.kwargs
        echograms_attrs = kwargs.get('echograms', {})
        enable_nc = echograms_attrs.get('enable_nc', False)
        enable_ascii = echograms_attrs.get('enable_ascii', False)
        enable_image = echograms_attrs.get('enable_image', False)

        # Initialize
        self._extras = pd.DataFrame()

        if enable_nc and enable_ascii:

            # Two possible outcomes:
            #     (1) If the echogram exists, align echometrics data along
            #         the echogram sample time.  There is a slight difference due to
            #         time being read with full float precision and reading time using
            #         the slocum binaries.  This should be less so if the dbdreader is
            #         utilized.
            #     (2) If the echogram does not exist, create an empty placeholder for
            #         the echogram and use the times available from the echometrics data.
            #
            # ECHOMETRIC_SENSOR and ECHOGRAM_VARS will have the extras dimension.
            # Any ECHOMETRIC_SENSOR variables will be removed from the data variable.

            # Initialize with empty dataframe until we confirm there is available data.
            echogram_data = pd.DataFrame(columns=ECHOGRAM_CSV_COLUMNS)

            if self.is_pq:
                # Read echogram from full precision data provided by dbdreader and
                # previously stored by parquet

                # Read echometrics and extract echogram
                have_teledyne = False
                try:
                    from gutils.slocum.echotools import teledyne
                    have_teledyne = True
                except ModuleNotFoundError:
                    L.error("Echotools teledyne python package missing.")

                if have_teledyne:
                    # Metadata options that can be defined
                    #echogramBins = echograms_attrs.get('echogram_range_bins', 20)
                    #echogramRange = echograms_attrs.get('echogram_range', 60.0)

                    segment, segExt = os.path.splitext(os.path.basename(self.ascii_file))

                    glider = teledyne.Glider()
                    glider.args = {}
                    glider.deployment = {
                        'extra_kwargs': kwargs
                    }
                    glider.updateArgumentsFromMetadata()

                    # Use Glider() framework to compute the echogram by
                    # faking original tbd and sbd inputs
                    glider.data['segment'] = segment
                    glider.data['input']['tbd'] = f'{segment}.tbd'
                    glider.data['input']['sbd'] = f'{segment}.sbd'

                    glider.data['columns']['sbd'] = [ 'm_present_time', 'm_depth']
                    glider.data['columns']['tbd'] = [
                        'sci_echodroid_aggindex', 'sci_echodroid_ctrmass',
                        'sci_echodroid_eqarea', 'sci_echodroid_inertia',
                        'sci_echodroid_propocc', 'sci_echodroid_sa',
                        'sci_echodroid_sv', 'sci_m_present_time']

                    # Copy data into glider.data as pandas and then to xarray
                    sbd = pd.DataFrame()
                    for col in glider.data['columns']['sbd']:
                        sbd[col] = self.data[col]
                    tbd = pd.DataFrame()
                    for col in glider.data['columns']['tbd']:
                        tbd[col] = self.data[col]

                    # Convert to xarray and assign
                    glider.data['sbd'] = sbd.to_xarray()
                    glider.data['tbd'] = tbd.to_xarray()

                    # Scan for metadata updates for this glider
                    #if echograms_attrs.get('svb_thresholds', None):
                    #    glider.args['vbsBins'] = echograms_attrs.get('svb_thresholds', [])

                    glider.readEchogram()
                    glider.createEchogramSpreadsheet()

                    # Convert glider.data['spreadsheet'] to echogram_data (if available)
                    echogram_data = pd.DataFrame()
                    if np.any(glider.data['spreadsheet']):
                        L.debug(f"Echogram read via parquet: {segment}")
                        for i in range(0, len(ECHOGRAM_CSV_COLUMNS)):
                            echogram_data[ECHOGRAM_CSV_COLUMNS[i]] =\
                                glider.data['spreadsheet'][:, i]
                        echogram_data['echogram_time'] = pd.to_datetime(
                            echogram_data['echogram_time'], unit='s', origin='unix'
                        )

                    # Determine if graphics need to be generated
                    if enable_image:
                        # Handle image: send to stdout or write to file
                        image_file = "segment_plottype.png"
                        glider.args['plotType'] = "binned,profile"
                        glider.args['outDir'] = os.path.dirname(self.ascii_file)
                        glider.args['imageOut'] = image_file
                        glider.args['title'] = "segment (plottype)"
                        glider.handleImage()
            else:
                egram_file = Path(self.ascii_file).with_suffix(".echogram")

                if egram_file.exists():
                    try:
                        echogram_data = pd.read_csv(
                            egram_file,
                            header=0,
                            names=ECHOGRAM_CSV_COLUMNS
                        )
                        echogram_data['echogram_time'] = pd.to_datetime(
                            echogram_data['echogram_time'], unit='s', origin='unix'
                        )
                    except BaseException as e:
                        L.warning(f"Could not process echogram file {egram_file}: {e}")

            # Do we have echometrics data?
            echometrics_data = pd.DataFrame()
            if 'sci_echodroid_sv' in data.columns:
                # Valid data rows are where Sv is less than 0 dB
                echometrics_data = data.loc[data['sci_echodroid_sv'] < 0, :]

            empty_echometrics_columns = {
                k: np.nan for k in ECHOMETRICS_SENSORS
            }
            empty_echogram_columns = {
                k: np.nan for k in ECHOGRAM_VARS
            }

            if not echogram_data.empty:

                # Create empty (nan) columns for the ECHOMETRICS variables
                self._extras = echogram_data.assign(**empty_echometrics_columns)

                if not echometrics_data.empty:

                    # Combine echogram and echometrics data together by matching with the first
                    # row that is within 30 seconds
                    for _, row in echometrics_data.iterrows():
                        idx = echogram_data.loc[
                            (echogram_data['echogram_time'] - row['t']).abs() < pd.Timedelta('30s')
                        ]
                        if not idx.empty:
                            self._extras.loc[
                                idx.iloc[0].name, ECHOMETRICS_SENSORS
                            ] = row.get(ECHOMETRICS_SENSORS)

                # Carry thought locations from the original glider data
                # since it was still moving while measuring the echometrics
                # and echograms
                self._extras = pd.merge_asof(
                    self._extras.set_index('echogram_time'),
                    data.set_index('t')[['x', 'y']],
                    left_index=True,
                    right_index=True,
                    direction='nearest',
                    tolerance=pd.Timedelta(seconds=60)
                ).reset_index()

                # Return a "standardized" dataframe with "t" as the index
                # and a column named "z".
                self._extras.rename(
                    columns={
                        'echogram_time': 't',
                        'echogram_depth': 'z'
                    },
                    inplace=True
                )

            elif not echometrics_data.empty:
                # NOTE: There is no test coverage here!
                echometrics_data.reset_index(inplace=True, drop=True)
                # Carry through the time and location of the data from the glider
                # There are captured as "extras" because this data would typically
                # be filtered out because it isn't an actual profile.
                self._extras = echometrics_data[ECHOMETRICS_SENSORS + ['t', 'x', 'y']]
                # If there is no echogram data, assign the echometrics data to z=0
                self._extras = self._extras.assign(
                    z=0.0,
                    **empty_echogram_columns
                )
            else:
                # Both data frames are empty, add empty columns
                self._extras = echogram_data.assign(
                    **{
                        **empty_echometrics_columns,
                        **empty_echogram_columns
                    }
                )

            # Once echometrics are copied out of data, the columns have to be removed from data
            # or the write to netCDF will fail due to duplicate variables.
            data = data.drop(columns=ECHOMETRICS_SENSORS, errors='ignore')
            data = data.drop(columns=ECHOGRAM_VARS, errors='ignore')

            if not self._extras.empty:
                self._extras = self._extras.sort_values(['t', 'z'])
                self._extras.set_index('t', inplace=True)

        return self._extras, data

    def read(self):

        # INIT
        metadata = OrderedDict()

        # PARQUET
        if self.is_pq:
            # PARQUET -> PANDAS
            read_ok = False
            if os.path.isfile(self.ascii_file):
                try:
                    df = pq.read_table(self.ascii_file).to_pandas()
                    read_ok = True
                except (pyarrow.ArrowInvalid):
                    L.error(f"Unable to read parquet file {self.ascii_file}")

            if not read_ok:
                # File does not exist, return empty objects
                return metadata, pd.DataFrame()

            # Reset "time" index and change back to "m_present_time"
            df = df.reset_index()
            df = df.rename(columns={"time": "m_present_time"})

            # Upstream wants:
            #   m_present_time/sci_m_present_time as dtype float64
            df['m_present_time'] = [(tm.value / 1000000000.0) for tm in df['m_present_time']]

            # Copy m_present_time to sci_m_present_time for upstream processing
            df['sci_m_present_time'] = df['m_present_time']

        else:
            # ASCII
            headers = None
            with open(self.ascii_file, 'rt') as af:
                for li, al in enumerate(af):
                    if 'm_present_time' in al:
                        headers = al.strip().split(' ')
                    elif headers is not None:
                        data_start = li + 2  # Skip units line and the integer row after that
                        break
                    else:
                        title, value = al.split(':', 1)
                        metadata[title.strip()] = value.strip()

            # Pull out the number of bytes for each column
            #   The last numerical field is the number of bytes transmitted for each sensor:
            #     1    A 1 byte integer value [-128 .. 127].
            #     2    A 2 byte integer value [-32768 .. 32767].
            #     4    A 4 byte float value (floating point, 6-7 significant digits,
            #                                approximately 10^-38 to 10^38 dynamic range).
            #     8    An 8 byte double value (floating point, 15-16 significant digits,
            #                                  approximately 10^-308 to 10^308 dyn. range).
            dtypedf = pd.read_csv(
                self.ascii_file,
                index_col=False,
                skiprows=data_start - 1,
                nrows=1,
                header=None,
                names=headers,
                sep=' ',
                skip_blank_lines=True,
            )

            #GUTILS/gutils/slocum/__init__.py:380: DeprecationWarning: `np.object` is a deprecated alias for
            #the builtin `object`. To silence this warning, use `object` by itself.
            #Doing this will not modify any behavior and is safe.
            #Deprecated in NumPy 1.20; for more details and guidance:
            #https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
            def intflag_to_dtype(intvalue):
                if intvalue == 1:
                    #return np.object  # ints can't have NaN so use object for now
                    return object  # ints can't have NaN so use object for now
                elif intvalue == 2:
                    #return np.object  # ints can't have NaN so use object for now
                    return object  # ints can't have NaN so use object for now
                elif intvalue == 4:
                    return np.float32
                elif intvalue == 8:
                    return np.float64
                else:
                    #return np.object
                    return object

            inttypes = [ intflag_to_dtype(x) for x in dtypedf.iloc[0].astype(int).values ]
            dtypes = dict(zip(dtypedf.columns, inttypes))

            df = pd.read_csv(
                self.ascii_file,
                index_col=False,
                skiprows=data_start,
                header=None,
                names=headers,
                dtype=dtypes,
                sep=' ',
                skip_blank_lines=True,
            )
            # m_present_time/sci_m_present_time are dtype float64

        return metadata, df

    def standardize(self, gps_prefix=None, z_axis_method=COMPUTE_PRESSURE):

        df = self.data.copy()

        # Convert NMEA coordinates to decimal degrees
        for col in df.columns:
            # Ignore if the m_gps_lat and/or m_gps_lon value is the default masterdata value
            if col.endswith('_lat'):
                df[col] = df[col].map(lambda x: get_decimal_degrees(x) if x <= 9000 else np.nan)
            elif col.endswith('_lon'):
                df[col] = df[col].map(lambda x: get_decimal_degrees(x) if x < 18000 else np.nan)

        # Standardize 'time' to the 't' column
        for t in self.TIMESTAMP_SENSORS:
            if t in df.columns:
                df['t'] = pd.to_datetime(df[t], unit='s')
                break

        # Interpolate GPS coordinates
        if 'm_gps_lat' in df.columns and 'm_gps_lon' in df.columns:

            df['drv_m_gps_lat'] = df.m_gps_lat.copy()
            df['drv_m_gps_lon'] = df.m_gps_lon.copy()

            # Fill in data will nulls where value is the default masterdata value
            masterdatas = (df.drv_m_gps_lon >= 18000) | (df.drv_m_gps_lat > 9000)
            df.loc[masterdatas, 'drv_m_gps_lat'] = np.nan
            df.loc[masterdatas, 'drv_m_gps_lon'] = np.nan

            try:
                # Interpolate the filled in 'x' and 'y'
                y_interp, x_interp = interpolate_gps(
                    masked_epoch(df.t),
                    df.drv_m_gps_lat,
                    df.drv_m_gps_lon
                )
            except (ValueError, IndexError):
                L.warning("Raw GPS values not found!")
                y_interp = np.empty(df.drv_m_gps_lat.size) * np.nan
                x_interp = np.empty(df.drv_m_gps_lon.size) * np.nan

            df['y'] = y_interp
            df['x'] = x_interp

        if z_axis_method == COMPUTE_PRESSURE:
            """
            ---- Option 1: Always calculate Z from pressure ----
            It's really a matter of data provider preference and varies from one provider to another.
            That being said, typically the sci_water_pressure or m_water_pressure variables, if present
            in the raw data files, will typically have more non-NaN values than m_depth.  For example,
            all MARACOOS gliders typically have both m_depth and sci_water_pressure contained in them.
            However, m_depth is typically heavily decimated while sci_water_pressure contains a more
            complete pressure record.  So, while we transmit both m_depth and sci_water_pressure, I
            calculate depth from pressure & (interpolated) latitude and use that as my NetCDF depth
            variable. - Kerfoot
            """
            # Search for a 'pressure' column
            for p in self.PRESSURE_SENSORS:
                if p in df.columns:
                    # Convert bar to dbar here
                    df['pressure'] = df[p].copy() * 10
                    # Calculate depth from pressure and latitude
                    # Negate the results so that increasing values note increasing depths
                    df['z'] = -z_from_p(df.pressure.values, df.y.values)
                    break

            if 'z' not in df and 'pressure' not in df:
                # Search for a 'z' column
                for p in self.DEPTH_SENSORS:
                    if p in df.columns:
                        df['z'] = df[p].copy()
                        # Calculate pressure from depth and latitude
                        # Negate the results so that increasing values note increasing depth
                        df['pressure'] = -p_from_z(df.z.values, df.y.values)
                        break

        elif z_axis_method == USE_RAW_PRESSURE:
            """
            ---- Option 2: Use raw pressure/depth data that was sent across ----
            """
            # Standardize to the 'pressure' column
            for p in self.PRESSURE_SENSORS:
                if p in df.columns:
                    # Convert bar to dbar here
                    df['pressure'] = df[p].copy() * 10
                    break

            # Standardize to the 'z' column
            for p in self.DEPTH_SENSORS:
                if p in df.columns:
                    df['z'] = df[p].copy()
                    break

            # Don't calculate Z from pressure if a metered depth column exists already
            if 'pressure' in df and 'z' not in df:
                # Calculate depth from pressure and latitude
                # Negate the results so that increasing values note increasing depths
                df['z'] = -z_from_p(df.pressure.values, df.y.values)

            if 'z' in df and 'pressure' not in df:
                # Calculate pressure from depth and latitude
                # Negate the results so that increasing values note increasing depth
                df['pressure'] = -p_from_z(df.z.values, df.y.values)

        else:
            raise ValueError("No z-axis method exists for {}".format(z_axis_method))

        rename_columns = {
            'm_water_vx': 'u_orig',
            'm_water_vy': 'v_orig',
        }

        # These need to be standardize so we can compute salinity and density!
        for vname in self.TEMPERATURE_SENSORS:
            if vname in df.columns:
                rename_columns[vname] = 'temperature'
                break
        for vname in self.CONDUCTIVITY_SENSORS:
            if vname in df.columns:
                rename_columns[vname] = 'conductivity'
                break

        # Standardize columns
        df = df.rename(columns=rename_columns)

        # Compute additional columns
        df = self.compute(df)

        return df

    def compute(self, df):
        try:
            # Compute salinity
            df['salinity'] = calculate_practical_salinity(
                conductivity=df.conductivity.values,
                temperature=df.temperature.values,
                pressure=df.pressure.values,
            )
        except (ValueError, AttributeError) as e:
            L.error("Could not compute salinity for {}: {}".format(self.ascii_file, e))

        try:
            # Compute density
            df['density'] = calculate_density(
                temperature=df.temperature.values,
                pressure=df.pressure.values,
                salinity=df.salinity.values,
                latitude=df.y.values,
                longitude=df.x.values,
            )
        except (ValueError, AttributeError) as e:
            L.error("Could not compute density for {}: {}".format(self.ascii_file, e))

        return df


class SlocumMerger(object):
    """
    Merges flight and science data files into an ASCII file.

    Copies files matching the regex in source_directory to their own temporary directory
    before processing since the Rutgers supported script only takes folders as input

    Returns a list of flight/science files that were processed into ASCII files
    """

    def __init__(self, source_directory, destination_directory, cache_directory=None, globs=None, deployments_path=None, template=None, prefer_file_filters=False, **filter_args):

        globs = globs or ['*']

        self.tmpdir = mkdtemp(prefix='gutils_convert_')
        self.matched_files = []
        self.cache_directory = cache_directory or source_directory
        self.destination_directory = destination_directory
        self.source_directory = source_directory

        mf = set()
        for g in globs:
            mf.update(
                glob(
                    os.path.join(
                        source_directory,
                        g
                    )
                )
            )

        def slocum_binary_sorter(x):
            """ Sort slocum binary files correctly, using leading zeros.leading """
            'usf-bass-2014-048-2-1.tbd -> 2014_048_00000002_000000001'
            x, ext = os.path.splitext(os.path.basename(x))
            if ext not in ALL_EXTENSIONS:
                return x
            z = [ int(a) for a in x.split('-')[-4:] ]
            return '{0[0]:04d}_{0[1]:03d}_{0[2]:08d}_{0[3]:08d}'.format(z)

        self.matched_files = sorted(list(mf), key=slocum_binary_sorter)

        # Initialize attrs as is done for gutils.nc.create_dataset()
        # attrs and extra_kwargs are stored in the object
        if deployments_path is None:
            self.attrs = {}
            self.filters = {}
            self.extra_kwargs = {}
            return

        # Use the first file as a place to start looking for the config directory
        try:
            file = self.matched_files[0]
        except BaseException:
            L.warning("No matched files")
            self.attrs = {}
            self.filters = {}
            self.extra_kwargs = {}
            return

        # Remove None filters from the arguments
        filter_args = { k: v for k, v in filter_args.items() if v is not None }

        # Figure out the ascii output path based on the file and the deployments_path
        dep_path = Path(deployments_path)
        file_path = Path(file)
        individual_dep_path = None
        for pp in file_path.parents:
            if dep_path == pp:
                break
            individual_dep_path = pp
        config_path = individual_dep_path / 'config'

        # Extract the filters from the config and override with passed in filters that are not None
        self.attrs = read_attrs(config_path, template=template)
        self.file_filters = self.attrs.pop('filters', {})

        # By default the filters passed in as filter_args will overwrite the filters defined in the
        # config file. If the opposite should happen (typically on a watch that uses a global set
        # of command line filters), you can set prefer_file_filters=True to have the file filters
        # take precedence over the passed in filters.
        if prefer_file_filters is False:
            self.filters = dict_update(self.file_filters, filter_args)
        else:
            self.filters = dict_update(filter_args, self.file_filters)

        # Kwargs can be defined in the "extra_kwargs" section of
        # a configuration object and passed into the extras method
        # of a reader.
        self.extra_kwargs = self.attrs.pop('extra_kwargs', {})

        if self.extra_kwargs.get('enable_debug', False):
            L.setLevel(logging.DEBUG)

    def __del__(self):
        # Remove tmpdir
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def block_read(self, dbd, duplicate_columns=dict()):
        """ Block read dbdreader information to pandas """

        blob = dbd.get(*dbd.parameterNames, return_nans=True, decimalLatLon=False)

        # dbdreader returns a list [] of returned columns
        # each list item is a tuple of (time, data)
        # Restack the data in a dict()

        data = {}
        for i in range(0, len(dbd.parameterNames)):
            data[dbd.parameterNames[i]] = blob[i][1]

        # Duplicate columns if requested
        for k, v in duplicate_columns.items():
            if v in data:
                data[k] = data[v]

        # Transform blob into pandas
        blob_data = pd.DataFrame.from_dict(data)

        return blob_data

    def read_header(self, dbdSource):
        """ Extract certain DBD header elements """

        keep_reading = True
        full_filename = ""
        filename_extension = ""
        is_dbd = ""
        cac = ""
        count = 0
        fn = open(dbdSource, 'rb')
        ln = ""
        while keep_reading:
            ch = fn.read(1).decode('utf-8', 'ignore')
            if ch == '\n':
                if ln:
                    if ln.startswith('dbd_label:'):
                        count += 1
                        data = ln.split(" ")
                        is_dbd = data[-1]
                    if ln.startswith('full_filename:'):
                        count += 1
                        data = ln.split(" ")
                        full_filename = data[-1]
                    if ln.startswith('filename_extension:'):
                        count += 1
                        data = ln.split(" ")
                        filename_extension = data[-1]
                    if ln.startswith('sensor_list_crc:'):
                        count += 1
                        data = ln.split(" ")
                        cac = data[-1]
                    ln = ""
                else:
                    keep_reading = False

                # All elements read
                if count == 4:
                    keep_reading = False
            else:
                if ch:
                    ln = ln + ch
                else:
                    keep_reading = False
        fn.close()
        return (full_filename, filename_extension, is_dbd, cac)

    def convert(self):
        # Convert DBDs: self.extra_kwargs
        #  (1) ascii: enable_parquet: false
        #  (2) dbdreader/parquet: enable_parquet: true

        # Make this user switchable to allow for graceful transition
        use_parquet = self.extra_kwargs.get('enable_parquet', False)

        # Copy to tempdir
        for f in self.matched_files:
            fname = os.path.basename(f)
            tmpf = os.path.join(self.tmpdir, fname)
            shutil.copy2(f, tmpf)

        safe_makedirs(self.destination_directory)

        # Returned from convert()
        processed = []

        if use_parquet:
            L.debug("PARQUET enabled deployment (SlocumMerger)")

            # self.tmpdir (source of DBD files)
            # self.destination_directory (ascii destination)

            # Processing pairs of files
            # SBD,sbd => TBD,tbd
            # MBD,mbd => NBD,nbd
            # DBD,dbd => EBD,ebd
            sciExt_lu = {
                'SBD': 'TBD',
                'MBD': 'NBD',
                'DBD': 'EBD',
                'sbd': 'tbd',
                'mdb': 'nbd',
                'dbd': 'ebd',
            }
            fli_keys = list(sciExt_lu.keys())

            dbdCount = 0
            dbdRoot = self.tmpdir
            for dbdSource in os.listdir(self.tmpdir):
                fliSource = os.path.join(self.tmpdir, dbdSource)

                # Files only
                if not os.path.isfile(fliSource):
                    continue

                # File must be of type data
                pargs = ["file", fliSource]
                command_output, return_code = generate_stream(pargs)
                if return_code == 0:
                    output = command_output.read().strip()
                    if not output.endswith('data'):
                        continue
                else:
                    continue

                # Strip off extension
                segment, dbdExt = os.path.splitext(dbdSource)

                # Drop leading period(.) in extension
                dbdExt = dbdExt.strip('.')

                # Get the real filename from the binary file header
                # Get the real extension from the binary file header
                dbdSeg, fType, is_dbd, cac = self.read_header(fliSource)

                if not is_dbd.startswith('DBD'):
                    L.error(f"Invalid flight source file: {dbdSource}")
                    continue

                # Determine the other file type to look for based on this dbdExtension
                if dbdExt in fli_keys:
                    sciExt = sciExt_lu[dbdExt]
                else:
                    continue

                dbdCount += 1

                sciSource = f"{dbdRoot}/{segment}.{sciExt}"
                pqExt = "pq"
                pqFile = os.path.join(self.destination_directory, f"{dbdSeg}.{pqExt}")
                datFile = os.path.join(self.destination_directory, f"{dbdSeg}.dat")

                msg = f"Creating {pqFile}"
                L.debug(msg)

                # Perform the decoding using dbdreader

                # Flight source
                if os.path.isfile(fliSource):
                    fli = dbdreader.DBD(fliSource, cacheDir=self.cache_directory, skip_initial_line=False)
                    fli_data = self.block_read(fli)
                    fli.close()
                    fli_data = fli_data.set_index('m_present_time')

                if os.path.isfile(sciSource):
                    sci = dbdreader.DBD(sciSource, cacheDir=self.cache_directory, skip_initial_line=False)
                    sci_data = self.block_read(sci, duplicate_columns={'m_present_time': 'sci_m_present_time'})
                    sci.close()
                    sci_data = sci_data.set_index('m_present_time')

                    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
                    dbd_data = pd.merge(fli_data, sci_data, on=['m_present_time'], how='outer', sort=True)
                else:
                    L.info("Converting flight data file ONLY")
                    dbd_data = fli_data

                dbd_data = dbd_data.reset_index()

                # drop sci_m_present_time and rename m_present_time to time
                if 'sci_m_present_time' in dbd_data.columns:
                    dbd_data = dbd_data.drop('sci_m_present_time', axis=1)

                dbd_data = dbd_data.rename(columns={"m_present_time": "time"})
                dbd_data['time'] = [datetime.datetime.fromtimestamp(tm, tz=datetime.timezone.utc) for tm in dbd_data['time'].values]
                dbd_data = dbd_data.set_index('time')
                dbd_data.to_parquet(pqFile)

                # upstream expects *.dat files, so produce
                # a *.pq file and link with to *.dat file
                try:
                    os.symlink(os.path.basename(pqFile), datFile)
                except FileExistsError:
                    L.error(f"Symlink already exists for {datFile}")
        else:
            # Run conversion script
            convert_binary_path = os.path.join(
                os.path.dirname(__file__),
                'bin',
                'convertDbds.sh'
            )
            pargs = [
                convert_binary_path,
                '-q',
                '-p',
                '-c', self.cache_directory
            ]

            echograms_attrs = self.extra_kwargs.get('echograms', {})
            enable_ascii = echograms_attrs.get('enable_ascii', False)
            enable_image = echograms_attrs.get('enable_image', False)
            enable_debug = self.extra_kwargs.get('enable_debug', False)

            if enable_debug:
                pargs.remove('-q')
                pargs.append('-D')

            if enable_ascii:
                # Perform echograms if this ASCII file matches the deployment
                # name of things we know to have the data. There needs to be a
                # better way to figure this out, but we don't have any understanding
                # of a deployment config object at this point.

                # Ideally this code isn't tacked into convertDbds.sh to output separate
                # files and can be done using the ASCII files exported from SlocumMerger
                # using pandas. Tighter integration into GUTILS will be done by
                # Rob Cermak@{UAF,UW}/John Horne@UW.  For now, the code is separate
                # and placed in # GUTILS/gutils/slocum/echotools.
                # The tools require one additional python library: dbdreader
                # https://github.com/smerckel/dbdreader

                # Defaults
                echogramBins = echograms_attrs.get('echogram_range_bins', 20)
                echogramRange = echograms_attrs.get('echogram_range', 60.0)
                vbsBins = echograms_attrs.get('svb_thresholds', [-34, -40, -46, -52, -58, -64, -70])
                dBLimits = echograms_attrs.get('svb_limits', [-30.0, -80.0])

                # Attempt to suss out the data type 'rt' or 'delayed' using
                # self.destination_directory.  Default to 'rt'.
                echogramType = 'rt'
                try:
                    dest_split_path = self.destination_directory.split('/')
                    foundType = dest_split_path[-2]
                    allowedEchogramTypes = ['sfmc', 'rt', 'delayed']
                    if foundType in allowedEchogramTypes:
                        echogramType = foundType
                except BaseException as e:
                    L.warning(f"Could not determine echogram data type: {e}")

                pargs = pargs + [
                    '-y', sys.executable,
                    '-g',  # Makes the echogram ASCII
                    '-t', f"{echogramType}",
                    '-r', f"{echogramRange}",
                    '-n', f"{echogramBins}",
                    '-V', f"{vbsBins}",
                    '-L', f"{dBLimits}"
                ]

                if enable_image:
                    echogramPlotType = echograms_attrs.get('plot_type', 'binned,profile')
                    echogramPlotCmap = echograms_attrs.get('plot_cmap', 'ek80')
                    pargs.append('-i')  # Makes the echogram images. This is slow!
                    pargs.append(f"{echogramPlotType}")
                    pargs.append('-C')
                    pargs.append(f"{echogramPlotCmap}")
                    L.debug(f"Creating echogram {echogramPlotType} images with colormap {echogramPlotCmap}")

            pargs.append(self.tmpdir)
            pargs.append(self.destination_directory)

            command_output, return_code = generate_stream(pargs)

            # Process returned files
            command_output_string = command_output.read()
            if enable_debug:
                L.setLevel(logging.DEBUG)
                L.debug(command_output_string)
            output_files = command_output_string.split('\n')

            # iterate and every time we hit a .dat file we return the cache
            binary_files = []
            for x in output_files:

                if x.startswith('Error'):
                    L.error(x)
                    continue

                if x.startswith('Skipping'):
                    continue

                fname = os.path.basename(x)
                _, suff = os.path.splitext(fname)

                if suff == '.dat':
                    ascii_file = os.path.join(self.destination_directory, fname)
                    if os.path.isfile(ascii_file):
                        processed.append({
                            'ascii': ascii_file,
                            'binary': sorted(binary_files)
                        })
                        L.info("Converted {} to {}".format(
                            ','.join([ os.path.basename(x) for x in sorted(binary_files) ]),
                            fname
                        ))
                    else:
                        L.warning("{} not an output file".format(x))

                    binary_files = []
                else:
                    bf = os.path.join(self.source_directory, fname)
                    if os.path.isfile(x):
                        binary_files.append(bf)

        return processed
