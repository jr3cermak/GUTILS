#!python
# coding=utf-8
import os
import time
import shutil
from glob import glob

from pyinotify import (
    IN_CLOSE_WRITE,
    IN_MOVED_TO,
    ThreadedNotifier,
    WatchManager
)

from gutils import safe_makedirs
from gutils.slocum import SlocumMerger
from gutils.watch.binary import Slocum2AsciiProcessor
from gutils.watch.ascii import Slocum2NetcdfProcessor
from gutils.watch.netcdf import Netcdf2ErddapProcessor
from gutils.tests import resource, output, GutilsTestClass

import logging
L = logging.getLogger(__name__)  # noqa

config_path = resource('slocum', 'config', 'bass-20160909T1733')
original_binary = resource('slocum', 'binary', 'bass-20160909T1733')
binary_path = output('binary', 'bass-20160909T1733')
ascii_path = output('ascii', 'bass-20160909T1733')
netcdf_path = output('netcdf', 'bass-20160909T1733')
erddap_content_path = output('erddap', 'content')
erddap_flag_path = output('erddap', 'flag')
ftp_path = output('ftp')


def wait_for_files(path, number):
        # Wait for NetCDF to be created
        count = 0
        loops = 20
        while True:
            try:
                num_files = len(os.listdir(path))
                assert num_files == number
                break
            except AssertionError:
                if count >= loops:
                    raise AssertionError("Not enough files in {}: Expected: {} Got: {}.".format(path, number, num_files))
                count += 1
                time.sleep(6)


class TestWatchClasses(GutilsTestClass):

    def setUp(self):
        super(TestWatchClasses, self).setUp()

        safe_makedirs(binary_path)
        safe_makedirs(ascii_path)
        safe_makedirs(netcdf_path)
        safe_makedirs(ftp_path)
        safe_makedirs(erddap_content_path)
        safe_makedirs(erddap_flag_path)

    def tearDown(self):
        shutil.rmtree(output())

    def test_gutils_binary_to_ascii_watch(self):

        wm = WatchManager()
        mask = IN_MOVED_TO | IN_CLOSE_WRITE

        # Convert binary data to ASCII
        processor = Slocum2AsciiProcessor(
            outputs_path=os.path.dirname(ascii_path)
        )
        notifier = ThreadedNotifier(wm, processor)
        notifier.coalesce_events()
        notifier.start()

        wdd = wm.add_watch(
            binary_path,
            mask,
            rec=True,
            auto_add=True
        )

        # Wait 5 seconds for the watch to start
        time.sleep(5)

        gpath = os.path.join(original_binary, '*.*bd')
        # Sort the files so the .cac files are generated in the right order
        for g in sorted(glob(gpath)):
            shutil.copy2(g, binary_path)

        wait_for_files(ascii_path, 32)

        wm.rm_watch(wdd.values(), rec=True)
        notifier.stop()

    def test_gutils_ascii_to_netcdf_watch(self):

        wm = WatchManager()
        mask = IN_MOVED_TO | IN_CLOSE_WRITE

        # Convert ASCII data to NetCDF
        processor = Slocum2NetcdfProcessor(
            outputs_path=os.path.dirname(netcdf_path),
            configs_path=os.path.dirname(config_path),
            subset=False,
            template='trajectory',
            profile_id_type=2,
            tsint=10,
            filter_distance=1,
            filter_points=5,
            filter_time=10,
            filter_z=1
        )
        notifier = ThreadedNotifier(wm, processor)
        notifier.coalesce_events()
        notifier.start()

        wdd = wm.add_watch(
            ascii_path,
            mask,
            rec=True,
            auto_add=True
        )

        # Wait 5 seconds for the watch to start
        time.sleep(5)

        # Make the ASCII we are watching for
        merger = SlocumMerger(
            original_binary,
            ascii_path,
            globs=['*.tbd', '*.sbd']
        )
        merger.convert()

        wait_for_files(netcdf_path, 230)

        wm.rm_watch(wdd.values(), rec=True)
        notifier.stop()

    def test_gutils_netcdf_to_erddap_watch(self):

        wm = WatchManager()
        mask = IN_MOVED_TO | IN_CLOSE_WRITE

        # Convert ASCII data to NetCDF
        processor = Netcdf2ErddapProcessor(
            outputs_path=os.path.dirname(netcdf_path),
            erddap_content_path=erddap_content_path,
            erddap_flag_path=erddap_flag_path
        )
        notifier = ThreadedNotifier(wm, processor, read_freq=5)
        notifier.coalesce_events()
        notifier.start()

        wdd = wm.add_watch(
            netcdf_path,
            mask,
            rec=True,
            auto_add=True
        )

        # Wait 5 seconds for the watch to start
        time.sleep(5)

        orig_netcdf = resource('profile.nc')
        dummy_netcdf = os.path.join(netcdf_path, 'profile.nc')
        shutil.copy(orig_netcdf, dummy_netcdf)

        wait_for_files(erddap_content_path, 1)
        wait_for_files(erddap_flag_path, 1)

        wm.rm_watch(wdd.values(), rec=True)
        notifier.stop()
