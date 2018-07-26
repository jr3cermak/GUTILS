#!/usr/bin/env python
import os
import shutil
from glob import glob
from tempfile import mkdtemp
from collections import OrderedDict

import numpy as np
import pandas as pd
from gsw import z_from_p, p_from_z

from gutils import (
    generate_stream,
    get_decimal_degrees,
    interpolate_gps,
    masked_epoch,
    safe_makedirs
)
from gutils.ctd import calculate_practical_salinity, calculate_density

import logging
L = logging.getLogger(__name__)


MODE_MAPPING = {
    "rt": ["sbd", "tbd", "mbd", "nbd"],
    "delayed": ["dbd", "ebd"]
}
ALL_EXTENSIONS = [".sbd", ".tbd", ".mbd", ".nbd", ".dbd", ".ebd"]


class SlocumReader(object):

    TIMESTAMP_SENSORS = ['m_present_time', 'sci_m_present_time']
    PRESSURE_SENSORS = ['sci_water_pressure', 'm_water_pressure', 'm_pressure']
    DEPTH_SENSORS = ['m_depth', 'm_water_depth']
    TEMPERATURE_SENSORS = ['sci_water_temp']
    CONDUCTIVITY_SENSORS = ['sci_water_cond']

    def __init__(self, ascii_file):
        self.ascii_file = ascii_file
        self.metadata, self.data = self.read()

        # Set the mode to 'rt' or 'delayed'
        self.mode = None
        if 'filename_extension' in self.metadata:
            filemode = self.metadata['filename_extension']
            for m, extensions in MODE_MAPPING.items():
                if filemode in extensions:
                    self.mode = m
                    break

    def read(self):
        metadata = OrderedDict()
        headers = None
        with open(self.ascii_file, 'rt') as af:
            for li, al in enumerate(af):
                if 'm_present_time' in al:
                    headers = al.strip().split(' ')
                elif headers is not None:
                    data_start = li + 2  # Skip units line and the interger row after that
                    break
                else:
                    title, value = al.split(':', 1)
                    metadata[title.strip()] = value.strip()

        df = pd.read_csv(
            self.ascii_file,
            index_col=False,
            skiprows=data_start,
            header=None,
            names=headers,
            sep=' ',
            skip_blank_lines=True,
        )
        return metadata, df

    def standardize(self, gps_prefix=None):

        df = self.data.copy()

        # Convert NMEA coordinates to decimal degrees
        for col in df.columns:
            # Ignore if the m_gps_lat and/or m_gps_lon value is the default masterdata value
            if '_lat' in col:
                df[col] = df[col].map(lambda x: get_decimal_degrees(x) if x <= 9000 else np.nan)
            elif '_lon' in col:
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
                df['z'] = -z_from_p(df.pressure, df.y)
                break

        if 'z' not in df and 'pressure' not in df:
            # Search for a 'z' column
            for p in self.DEPTH_SENSORS:
                if p in df.columns:
                    df['z'] = df[p].copy()
                    # Calculate pressure from depth and latitude
                    # Negate the results so that increasing values note increasing depth
                    df['pressure'] = -p_from_z(df.z, df.y)
                    break
        # End Option 1

        """
        ---- Option 2: Use raw pressure/depth data that was sent across ----
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
            df['z'] = -z_from_p(df.pressure, df.y)

        if 'z' in df and 'pressure' not in df:
            # Calculate pressure from depth and latitude
            # Negate the results so that increasing values note increasing depth
            df['pressure'] = -p_from_z(df.z, df.y)
        # End Option 2
        """

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
                latitude=df.y,
                longitude=df.x,
            )
        except (ValueError, AttributeError) as e:
            L.error("Could not compute density for {}: {}".format(self.ascii_file, e))

        return df


class SlocumMerger(object):
    """
    Merges flight and science data files into an ASCII file.

    Copies files matching the regex in source_directory to their own temporary directory
    before processing since the Rutgers supported script only takes foldesr as input

    Returns a list of flight/science files that were processed into ASCII files
    """

    def __init__(self, source_directory, destination_directory, cache_directory=None, globs=None):

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

    def __del__(self):
        # Remove tmpdir
        shutil.rmtree(self.tmpdir)

    def convert(self):
        # Copy to tempdir
        for f in self.matched_files:
            fname = os.path.basename(f)
            tmpf = os.path.join(self.tmpdir, fname)
            shutil.copy2(f, tmpf)

        safe_makedirs(self.destination_directory)

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

        pargs.append(self.tmpdir)
        pargs.append(self.destination_directory)

        command_output, return_code = generate_stream(pargs)

        # Return
        processed = []
        output_files = command_output.read().split('\n')
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
