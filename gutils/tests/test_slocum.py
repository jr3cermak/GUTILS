#!python
# coding=utf-8
import os
import shutil
import tempfile
from glob import glob
from collections import namedtuple

import pytest
import netCDF4 as nc4
from gutils.slocum import SlocumMerger, SlocumReader
from gutils.tests import GutilsTestClass, resource
from gutils.nc import check_dataset, create_dataset

import logging
L = logging.getLogger(__name__)  # noqa


class TestSlocumMerger(GutilsTestClass):

    def setUp(self):
        super().setUp()
        self.binary_path = resource('slocum', 'bass-20150407T1300', 'rt', 'binary')
        self.ascii_path = resource('slocum', 'bass-20150407T1300', 'rt', 'ascii')

    def tearDown(self):
        shutil.rmtree(self.ascii_path, ignore_errors=True)  # Remove generated ASCII
        # Remove any cached .cac files
        for cac in glob(os.path.join(self.binary_path, '*.cac')):
            os.remove(cac)

    def test_convert_default_cache_directory(self):
        merger = SlocumMerger(
            self.binary_path,
            self.ascii_path,
            globs=['*.tbd', '*.sbd']
        )
        p = merger.convert()
        assert len(p) > 0
        assert len(glob(os.path.join(self.ascii_path, '*.dat'))) > 0

    def test_convert_empty_cache_directory(self):
        merger = SlocumMerger(
            self.binary_path,
            self.ascii_path,
            cache_directory=tempfile.mkdtemp(),
            globs=['*.tbd', '*.sbd']
        )
        p = merger.convert()
        assert len(p) > 0
        assert len(glob(os.path.join(self.ascii_path, '*.dat'))) > 0

    def test_convert_single_pair(self):
        merger = SlocumMerger(
            self.binary_path,
            self.ascii_path,
            globs=['usf-bass-2014-048-0-0.tbd', 'usf-bass-2014-048-0-0.sbd']
        )
        p = merger.convert()
        assert p == [{
            'ascii': os.path.join(self.ascii_path, 'usf_bass_2014_048_0_0_sbd.dat'),
            'binary': sorted([
                os.path.join(self.binary_path, 'usf-bass-2014-048-0-0.sbd'),
                os.path.join(self.binary_path, 'usf-bass-2014-048-0-0.tbd')
            ]),
        }]
        assert len(glob(os.path.join(self.ascii_path, '*.dat'))) == 1

        af = p[0]['ascii']
        sr = SlocumReader(af)
        raw = sr.data
        assert 'density' not in raw.columns
        assert 'salinity' not in raw.columns
        assert 't' not in raw.columns
        assert 'x' not in raw.columns
        assert 'y' not in raw.columns
        assert 'z' not in raw.columns

        enh = sr.standardize()
        assert not enh['density'].any()  # No GPS data so we can't compute density
        assert 'salinity' in enh.columns
        assert 't' in enh.columns
        assert 'x' in enh.columns
        assert 'y' in enh.columns
        assert 'z' in enh.columns


class TestSlocumReaderWithGPS(GutilsTestClass):

    def setUp(self):
        super().setUp()
        self.binary_path = resource('slocum', 'bass-20160909T1733', 'rt', 'binary')
        self.ascii_path = resource('slocum', 'bass-20160909T1733', 'rt', 'ascii')

    def tearDown(self):
        shutil.rmtree(self.ascii_path, ignore_errors=True)  # Remove generated ASCII
        # Remove any cached .cac files
        for cac in glob(os.path.join(self.binary_path, '*.cac')):
            os.remove(cac)

    def test_read_all_pairs_gps(self):
        merger = SlocumMerger(
            self.binary_path,
            self.ascii_path,
            globs=['*.tbd', '*.sbd']
        )
        p = merger.convert()
        af = p[0]['ascii']

        sr = SlocumReader(af)
        raw = sr.data
        assert 'density' not in raw.columns
        assert 'salinity' not in raw.columns
        assert 't' not in raw.columns
        assert 'x' not in raw.columns
        assert 'y' not in raw.columns
        assert 'z' not in raw.columns

        enh = sr.standardize()
        assert 'density' in enh.columns
        assert 'salinity' in enh.columns
        assert 't' in enh.columns
        assert 'x' in enh.columns
        assert 'y' in enh.columns
        assert 'z' in enh.columns


@pytest.mark.long
class TestSlocumExportDelayed(GutilsTestClass):

    def setUp(self):
        super().setUp()
        self.binary_path = resource('slocum', 'modena-20150625T0000', 'delayed', 'binary')
        self.ascii_path = resource('slocum', 'modena-20150625T0000', 'delayed', 'ascii')
        self.cache_path = resource('slocum', 'modena-20150625T0000', 'config')

    def tearDown(self):
        shutil.rmtree(self.ascii_path, ignore_errors=True)  # Remove generated ASCII

    def test_single_pair_existing_cac_files(self):
        # The 0 files are there to produce the required .cac files
        merger = SlocumMerger(
            self.binary_path,
            self.ascii_path,
            cache_directory=self.cache_path,
            globs=['modena-2015-175-0-9.dbd', 'modena-2015-175-0-9.ebd']
        )
        p = merger.convert()
        af = p[-1]['ascii']

        sr = SlocumReader(af)
        raw = sr.data
        assert 'density' not in raw.columns
        assert 'salinity' not in raw.columns
        assert 't' not in raw.columns
        assert 'x' not in raw.columns
        assert 'y' not in raw.columns
        assert 'z' not in raw.columns

        enh = sr.standardize()
        assert 'density' in enh.columns
        assert 'salinity' in enh.columns
        assert 't' in enh.columns
        assert 'x' in enh.columns
        assert 'y' in enh.columns
        assert 'z' in enh.columns


@pytest.mark.long
class TestEchodroidOne(GutilsTestClass):

    def setUp(self):
        super().setUp()
        self.binary_path = resource('slocum', 'unit_507_one', 'delayed', 'binary')
        self.ascii_path = resource('slocum', 'unit_507_one', 'delayed', 'ascii')
        self.netcdf_path = resource('slocum', 'unit_507_one', 'delayed', 'netcdf')
        self.cache_path = resource('slocum', 'unit_507_one', 'config')

    def tearDown(self):
        shutil.rmtree(self.ascii_path, ignore_errors=True)  # Remove generated ASCII
        shutil.rmtree(self.netcdf_path, ignore_errors=True)  # Remove generated netCDF

    def test_echogram(self):
        merger = SlocumMerger(
            self.binary_path,
            self.ascii_path,
            cache_directory=self.cache_path,
            globs=['unit_507-2021-308*']
        )
        _ = merger.convert()

        dat_files = [ p for p in os.listdir(self.ascii_path) if p.endswith('.dat')]
        for ascii_file in dat_files:
            args = dict(
                file=os.path.join(self.ascii_path, ascii_file),
                reader_class=SlocumReader,
                deployments_path=resource('slocum'),
                subset=True,
                template='slocum_dac',
                profile_id_type=1,
                tsint=10,
                filter_distance=1,
                filter_points=5,
                filter_time=10,
                filter_z=1,
                z_axis_method=2
            )
            create_dataset(**args)

        assert os.path.exists(self.netcdf_path)

        output_files = sorted(os.listdir(self.netcdf_path))
        output_files = [ os.path.join(self.netcdf_path, o) for o in output_files ]
        assert len(output_files) == 28

        # First profile
        with nc4.Dataset(output_files[0]) as ncd:
            assert ncd.variables['profile_id'].ndim == 0
            assert ncd.variables['profile_id'][0] == 1636072703

        # Last profile
        with nc4.Dataset(output_files[-1]) as ncd:
            assert ncd.variables['profile_id'].ndim == 0
            assert ncd.variables['profile_id'][0] == 1636146248

        # Check netCDF file for compliance
        ds = namedtuple('Arguments', ['file'])
        for o in output_files:
            assert check_dataset(ds(file=o)) == 0


@pytest.mark.long
class TestEchodroidTwo(GutilsTestClass):

    def setUp(self):
        super().setUp()
        self.binary_path = resource('slocum', 'unit_507_two', 'rt', 'binary')
        self.ascii_path = resource('slocum', 'unit_507_two', 'rt', 'ascii')
        self.netcdf_path = resource('slocum', 'unit_507_two', 'rt', 'netcdf')
        self.cache_path = resource('slocum', 'unit_507_two', 'config')

    def tearDown(self):
        shutil.rmtree(self.ascii_path, ignore_errors=True)  # Remove generated ASCII
        shutil.rmtree(self.netcdf_path, ignore_errors=True)  # Remove generated netCDF

    def test_echogram(self):
        merger = SlocumMerger(
            self.binary_path,
            self.ascii_path,
            cache_directory=self.cache_path,
            globs=['unit_507-2021-*']
        )
        _ = merger.convert()

        dat_files = [ p for p in os.listdir(self.ascii_path) if p.endswith('.dat')]
        for ascii_file in dat_files:
            args = dict(
                file=os.path.join(self.ascii_path, ascii_file),
                reader_class=SlocumReader,
                deployments_path=resource('slocum'),
                subset=True,
                template='slocum_dac',
                profile_id_type=1,
                tsint=10,
                filter_distance=1,
                filter_points=5,
                filter_time=10,
                filter_z=1,
                z_axis_method=1
            )
            create_dataset(**args)

        assert os.path.exists(self.netcdf_path)

        output_files = sorted(os.listdir(self.netcdf_path))
        output_files = [ os.path.join(self.netcdf_path, o) for o in output_files ]
        assert len(output_files) == 29

        # First profile
        with nc4.Dataset(output_files[0]) as ncd:
            assert ncd.variables['profile_id'].ndim == 0
            assert ncd.variables['profile_id'][0] == 1638999396

        # Last profile
        with nc4.Dataset(output_files[-1]) as ncd:
            assert ncd.variables['profile_id'].ndim == 0
            assert ncd.variables['profile_id'][0] == 1639069272


@pytest.mark.long
class TestEchoMetricsOne(GutilsTestClass):

    def setUp(self):
        super().setUp()
        self.binary_path = resource('slocum', 'echometrics', 'rt', 'binary')
        self.ascii_path = resource('slocum', 'echometrics', 'rt', 'ascii')
        self.netcdf_path = resource('slocum', 'echometrics', 'rt', 'netcdf')
        self.cache_path = resource('slocum', 'echometrics', 'config')

    def tearDown(self):
        shutil.rmtree(self.ascii_path, ignore_errors=True)  # Remove generated ASCII
        shutil.rmtree(self.netcdf_path, ignore_errors=True)  # Remove generated netCDF

    def test_echogram(self):
        merger = SlocumMerger(
            self.binary_path,
            self.ascii_path,
            cache_directory=self.cache_path,
            globs=['*'],
            deployments_path=resource('slocum'),
            template='slocum_dac'
        )
        _ = merger.convert()

        dat_files = [ p for p in os.listdir(self.ascii_path) if p.endswith('.dat')]
        for ascii_file in dat_files:
            args = dict(
                file=os.path.join(self.ascii_path, ascii_file),
                reader_class=SlocumReader,
                deployments_path=resource('slocum'),
                subset=True,
                template='slocum_dac',
                profile_id_type=1,
                tsint=10,
                filter_distance=1,
                filter_points=5,
                filter_time=10,
                filter_z=1,
                z_axis_method=1
            )
            create_dataset(**args)

        assert os.path.exists(self.netcdf_path)

        output_files = sorted(os.listdir(self.netcdf_path))
        output_files = [ os.path.join(self.netcdf_path, o) for o in output_files ]
        assert len(output_files) == 1

        # First profile
        with nc4.Dataset(output_files[0]) as ncd:
            assert ncd.variables['profile_id'].ndim == 0
            assert ncd.variables['profile_id'][0] == 1642638468

        # Last profile
        with nc4.Dataset(output_files[-1]) as ncd:
            assert ncd.variables['profile_id'].ndim == 0
            assert ncd.variables['profile_id'][0] == 1642638468

        # Check netCDF file for compliance
        ds = namedtuple('Arguments', ['file'])
        for o in output_files:
            assert check_dataset(ds(file=o)) == 0


@pytest.mark.long
class TestEchoMetricsTwo(GutilsTestClass):

    def setUp(self):
        super().setUp()
        self.binary_path = resource('slocum', 'echometrics2', 'rt', 'binary')
        self.ascii_path = resource('slocum', 'echometrics2', 'rt', 'ascii')
        self.netcdf_path = resource('slocum', 'echometrics2', 'rt', 'netcdf')
        self.cache_path = resource('slocum', 'echometrics2', 'config')

    def tearDown(self):
        shutil.rmtree(self.ascii_path, ignore_errors=True)  # Remove generated ASCII
        shutil.rmtree(self.netcdf_path, ignore_errors=True)  # Remove generated netCDF

    def test_echogram(self):
        merger = SlocumMerger(
            self.binary_path,
            self.ascii_path,
            cache_directory=self.cache_path,
            globs=['*'],
            deployments_path=resource('slocum'),
            template='slocum_dac'
        )
        _ = merger.convert()

        dat_files = [ p for p in os.listdir(self.ascii_path) if p.endswith('.dat')]
        for ascii_file in dat_files:
            args = dict(
                file=os.path.join(self.ascii_path, ascii_file),
                reader_class=SlocumReader,
                deployments_path=resource('slocum'),
                subset=True,
                template='slocum_dac',
                profile_id_type=1,
                tsint=10,
                filter_distance=1,
                filter_points=5,
                filter_time=10,
                filter_z=1,
                z_axis_method=1
            )
            create_dataset(**args)

        print(self.netcdf_path)
        assert os.path.exists(self.netcdf_path)

        output_files = sorted(os.listdir(self.netcdf_path))
        output_files = [ os.path.join(self.netcdf_path, o) for o in output_files ]
        assert len(output_files) == 2

        # First profile
        with nc4.Dataset(output_files[0]) as ncd:
            assert ncd.variables['profile_id'].ndim == 0
            # first time in the first profile
            assert ncd.variables['profile_id'][0] == 1679577217

        # Last profile
        with nc4.Dataset(output_files[-1]) as ncd:
            assert ncd.variables['profile_id'].ndim == 0
            # first time in the last ecodroid profile
            assert ncd.variables['profile_id'][0] == 1679577298

        # Check netCDF file for compliance
        ds = namedtuple('Arguments', ['file'])
        for o in output_files:
            assert check_dataset(ds(file=o)) == 0

        # Echogram check
        output_files = {
            '.dat': 0,
            '.echogram': 0,
            '.png': 0
        }
        output_file_keys = output_files.keys()
        for o in sorted(os.listdir(self.ascii_path)):
            _, ext = os.path.splitext(o)
            if ext in output_file_keys:
                output_files[ext] += 1
        assert output_files['.dat'] == 1
        assert output_files['.png'] == 1
        assert output_files['.echogram'] == 1

        # Passive test 'teledyne' module from echotools
        try:
            import glob
            import sys
            sys.path.append('gutils/slocum/echotools')
            import pandas as pd
            import teledyne

            # Setup the glider object
            glider = teledyne.Glider()

            # Debug control: set True to increase verbosity
            glider.debugFlag = False

            segment = 'unit_507-2023-079-4-46'
            args = {
                'deploymentDir': self.cache_path,
                'templateDir': self.cache_path,
                'template': 'slocum_dac.json',
                'inpDir': self.binary_path,
                'cacheDir': self.cache_path
            }

            # Pass arguments to the glider class object
            glider.args = args

            glider.loadMetadata()
            glider.args = glider.updateArgumentsFromMetadata()

            # Pass arguments to the glider class object
            glider.args = args

            # Find and read glider files
            file_glob = os.path.join(args['inpDir'], "%s%s" % (segment, ".?[bB][dD]"))

            glider_files = glob.glob(file_glob)
            for glider_file in glider_files:
                glider.readDbd(
                    inputFile = glider_file,
                    cacheDir = args['cacheDir']
                )

            # Attempt to extract echogram from glider data
            glider.readEchogram()
            glider.createEchogramSpreadsheet()

            # The echogram is stored in a numpy data object
            echogram_numpy = glider.data['spreadsheet']

            #print("The echogram at this point is available in a numpy object")
            #print("with the columns in the first row: [time, depth, Sv]")
            #print(echogram_numpy[0, :])
            #print()
            assert echogram_numpy[0, 0] == 1679577318.4577026
            assert echogram_numpy[0, 1] == 245.7657470703125
            assert echogram_numpy[0, 2] == -73.0

            # Convert the echogram numpy object to pandas and set the time
            # column as an index.
            echogram_pandas = pd.DataFrame(data = echogram_numpy, columns = ['time', 'depth', 'Sv']).set_index('time')

            #print("The first row of the pandas data frame:")
            #print(echogram_pandas.iloc[0])
            #print()

            # Convert the pandas data frame to xarray, add units
            # and convert time timestamp to datetime object
            echogram_xarray = echogram_pandas.to_xarray()
            echogram_xarray['time'].attrs['units'] = 'seconds since 1970-01-01'
            echogram_xarray['time'] = pd.to_datetime(echogram_xarray['time'], unit='s')
            echogram_xarray['depth'].attrs['units'] = 'meters'
            echogram_xarray['Sv'].attrs['units'] = 'dB re 1 m2/m3'

            #print("The first row of the xarray data frame:")
            #print(echogram_xarray.isel(time=0))

        except BaseException as e:
            print("WARNING: echotools `teledyne` test silently failed.")
            print(e)


@pytest.mark.long
class TestEchoMetricsThree(GutilsTestClass):

    def setUp(self):
        super().setUp()
        self.binary_path = resource('slocum', 'echometrics3', 'rt', 'binary')
        self.ascii_path = resource('slocum', 'echometrics3', 'rt', 'ascii')
        self.netcdf_path = resource('slocum', 'echometrics3', 'rt', 'netcdf')
        self.cache_path = resource('slocum', 'echometrics3', 'config')

    def tearDown(self):
        shutil.rmtree(self.ascii_path, ignore_errors=True)  # Remove generated ASCII
        shutil.rmtree(self.netcdf_path, ignore_errors=True)  # Remove generated netCDF

    def test_echogram(self):
        merger = SlocumMerger(
            self.binary_path,
            self.ascii_path,
            cache_directory=self.cache_path,
            globs=['*'],
            deployments_path=resource('slocum'),
            template='slocum_dac'
        )
        _ = merger.convert()

        dat_files = [ p for p in os.listdir(self.ascii_path) if p.endswith('.dat')]
        for ascii_file in dat_files:
            args = dict(
                file=os.path.join(self.ascii_path, ascii_file),
                reader_class=SlocumReader,
                deployments_path=resource('slocum'),
                subset=True,
                template='slocum_dac',
                profile_id_type=1,
                z_axis_method=1
            )
            create_dataset(**args)

        assert os.path.exists(self.netcdf_path)

        output_files = sorted(os.listdir(self.netcdf_path))
        output_files = [ os.path.join(self.netcdf_path, o) for o in output_files ]
        assert len(output_files) == 3

        # First profile
        with nc4.Dataset(output_files[0]) as ncd:
            assert ncd.variables['profile_id'].ndim == 0
            assert ncd.variables['profile_id'][0] == 1644647093

        # Last profile
        with nc4.Dataset(output_files[-1]) as ncd:
            assert ncd.variables['profile_id'].ndim == 0
            assert ncd.variables['profile_id'][0] == 1644648114

        # Check netCDF file for compliance
        ds = namedtuple('Arguments', ['file'])
        for o in output_files:
            assert check_dataset(ds(file=o)) == 0


@pytest.mark.long
class TestEchoMetricsFour(GutilsTestClass):

    def setUp(self):
        super().setUp()
        self.binary_path = resource('slocum', 'echometrics4', 'rt', 'binary')
        self.ascii_path = resource('slocum', 'echometrics4', 'rt', 'ascii')
        self.netcdf_path = resource('slocum', 'echometrics4', 'rt', 'netcdf')
        self.cache_path = resource('slocum', 'echometrics4', 'config')

    def tearDown(self):
        shutil.rmtree(self.ascii_path, ignore_errors=True)  # Remove generated ASCII
        shutil.rmtree(self.netcdf_path, ignore_errors=True)  # Remove generated netCDF

    def test_echogram(self):
        merger = SlocumMerger(
            self.binary_path,
            self.ascii_path,
            cache_directory=self.cache_path,
            globs=['*'],
            deployments_path=resource('slocum'),
            template='slocum_dac'
        )
        _ = merger.convert()

        dat_files = [ p for p in os.listdir(self.ascii_path) if p.endswith('.dat')]
        for ascii_file in dat_files:
            args = dict(
                file=os.path.join(self.ascii_path, ascii_file),
                reader_class=SlocumReader,
                deployments_path=resource('slocum'),
                subset=True,
                template='slocum_dac',
                profile_id_type=1,
                z_axis_method=1
            )
            create_dataset(**args)

        assert os.path.exists(self.netcdf_path)
        assert os.path.exists(self.ascii_path)

        # Check number of profiles in netcdf directory
        output_files = sorted(os.listdir(self.netcdf_path))
        output_files = [ os.path.join(self.netcdf_path, o) for o in output_files ]
        assert len(output_files) == 12

        # First profile
        with nc4.Dataset(output_files[0]) as ncd:
            assert ncd.variables['profile_id'].ndim == 0
            assert ncd.variables['profile_id'][0] == 1647110555

        # Last profile
        with nc4.Dataset(output_files[-1]) as ncd:
            assert ncd.variables['profile_id'].ndim == 0
            assert ncd.variables['profile_id'][0] == 1647141015

        # Check netCDF file for compliance
        ds = namedtuple('Arguments', ['file'])
        for o in output_files:
            assert check_dataset(ds(file=o)) == 0

        # Echogram check
        output_files = {
            '.dat': 0,
            '.echogram': 0,
            '.png': 0
        }
        output_file_keys = output_files.keys()
        for o in sorted(os.listdir(self.ascii_path)):
            _, ext = os.path.splitext(o)
            if ext in output_file_keys:
                output_files[ext] += 1
        assert output_files['.dat'] == 10
        assert output_files['.png'] == 0
        assert output_files['.echogram'] == 0


@pytest.mark.long
class TestEchoMetricsFive(GutilsTestClass):

    def setUp(self):
        super().setUp()
        self.binary_path = resource('slocum', 'echometrics5', 'rt', 'binary')
        self.ascii_path = resource('slocum', 'echometrics5', 'rt', 'ascii')
        self.netcdf_path = resource('slocum', 'echometrics5', 'rt', 'netcdf')
        self.cache_path = resource('slocum', 'echometrics5', 'config')

    def tearDown(self):
        shutil.rmtree(self.ascii_path, ignore_errors=True)  # Remove generated ASCII
        shutil.rmtree(self.netcdf_path, ignore_errors=True)  # Remove generated netCDF

    def test_echogram(self):
        merger = SlocumMerger(
            self.binary_path,
            self.ascii_path,
            cache_directory=self.cache_path,
            globs=['*'],
            deployments_path=resource('slocum'),
            template='slocum_dac'
        )
        _ = merger.convert()

        dat_files = [ p for p in os.listdir(self.ascii_path) if p.endswith('.dat')]
        for ascii_file in dat_files:
            args = dict(
                file=os.path.join(self.ascii_path, ascii_file),
                reader_class=SlocumReader,
                deployments_path=resource('slocum'),
                subset=True,
                template='slocum_dac',
                profile_id_type=1,
                z_axis_method=1
            )
            create_dataset(**args)

        assert os.path.exists(self.netcdf_path)
        assert os.path.exists(self.ascii_path)

        output_files = sorted(os.listdir(self.netcdf_path))
        output_files = [ os.path.join(self.netcdf_path, o) for o in output_files ]

        for f in output_files:
            with nc4.Dataset(f) as ncd:
                assert 'time' in ncd.variables
                assert 'depth' in ncd.variables
                assert 'lat' in ncd.variables
                assert 'lon' in ncd.variables
                assert 'salinity' in ncd.variables
                assert 'pressure' in ncd.variables
                assert 'temperature' in ncd.variables
                assert 'profile_time' in ncd.variables
                assert 'profile_lat' in ncd.variables
                assert 'profile_lon' in ncd.variables
                assert 'profile_id' in ncd.variables
                assert 'sci_echodroid_aggindex' in ncd.variables
                assert 'sci_echodroid_ctrmass' in ncd.variables
                assert 'sci_echodroid_eqarea' in ncd.variables
                assert 'sci_echodroid_inertia' in ncd.variables
                assert 'sci_echodroid_propocc' in ncd.variables
                assert 'sci_echodroid_sa' in ncd.variables
                assert 'sci_echodroid_sv' in ncd.variables
                assert 'echogram_sv' in ncd.variables

        # Echogram check
        output_files = {
            '.dat': 0,
            '.echogram': 0,
            '.png': 0
        }
        output_file_keys = output_files.keys()
        for o in sorted(os.listdir(self.ascii_path)):
            _, ext = os.path.splitext(o)
            if ext in output_file_keys:
                output_files[ext] += 1
        assert output_files['.dat'] == 10
        assert output_files['.png'] == 5
        assert output_files['.echogram'] == 5
