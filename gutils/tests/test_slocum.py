#!python
# coding=utf-8
import os
import shutil
import tempfile
from glob import glob

from gutils.slocum import SlocumMerger, SlocumReader
from gutils.tests import GutilsTestClass, resource

import logging
L = logging.getLogger(__name__)  # noqa


class TestSlocumMerger(GutilsTestClass):

    def setUp(self):
        super(TestSlocumMerger, self).setUp()

        self.binary_path = resource('slocum', 'binary', 'bass-20150407T1300')
        self.ascii_path = resource('slocum', 'ascii', 'bass-20150407T1300')

    def tearDown(self):
        shutil.rmtree(self.ascii_path)  # Remove generated ASCII
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


class TestSlocumReaderNoGPS(GutilsTestClass):

    def setUp(self):
        super(TestSlocumReaderNoGPS, self).setUp()
        self.binary_path = resource('slocum', 'binary', 'bass-20150407T1300')
        self.ascii_path = resource('slocum', 'ascii', 'bass-20150407T1300')

    def tearDown(self):
        shutil.rmtree(self.ascii_path)  # Remove generated ASCII

        # Remove any cached .cac files
        for cac in glob(os.path.join(self.binary_path, '*.cac')):
            os.remove(cac)

    def test_read_single_pair(self):
        merger = SlocumMerger(
            self.binary_path,
            self.ascii_path,
            globs=['usf-bass-2014-048-0-0.tbd', 'usf-bass-2014-048-0-0.sbd']
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
        assert 'density' not in enh.columns  # No GPS data so we can't compute density
        assert 'salinity' in enh.columns
        assert 't' in enh.columns
        assert 'x' in enh.columns
        assert 'y' in enh.columns
        assert 'z' in enh.columns


class TestSlocumReaderWithGPS(GutilsTestClass):

    def setUp(self):
        super(TestSlocumReaderWithGPS, self).setUp()
        self.binary_path = resource('slocum', 'binary', 'bass-20160909T1733')
        self.ascii_path = resource('slocum', 'ascii', 'bass-20160909T1733')

    def tearDown(self):
        shutil.rmtree(self.ascii_path)  # Remove generated ASCII
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


class TestSlocumExportDelayed(GutilsTestClass):

    def setUp(self):
        super(TestSlocumExportDelayed, self).setUp()
        self.binary_path = resource('slocum', 'binary', 'modena-20150625T0000')
        self.ascii_path = resource('slocum', 'ascii', 'modena-20150625T0000')
        self.cache_path = resource('slocum', 'config', 'modena-20150625T0000')

    def tearDown(self):
        shutil.rmtree(self.ascii_path)  # Remove generated ASCII

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
