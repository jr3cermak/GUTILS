#!python
# coding=utf-8
import os
import shutil
from glob import glob

import pytest

from gutils.nc import create_dataset
from gutils.slocum import SlocumMerger, SlocumReader
from gutils.tests import setup_testing_logger, resource

import logging
L = logging.getLogger(__name__)  # noqa


@pytest.mark.long
@pytest.mark.parametrize("deployment", [
    ('unit_191-20210323T0000', 'rt'),
])
def test_real_deployments(deployment):
    setup_testing_logger(level=logging.WARNING)
    binary_path     = resource('slocum', deployment[0], deployment[1], 'binary')
    ascii_path      = resource('slocum', deployment[0], deployment[1], 'ascii')
    netcdf_path     = resource('slocum', deployment[0], deployment[1], 'netcdf')
    config_path     = resource('slocum', deployment[0], 'config')

    # Static args
    args = dict(
        reader_class=SlocumReader,
        deployments_path=resource('slocum'),
        subset=True,
        template='slocum_dac',
        profile_id_type=1,
        filter_distance=1,
        filter_points=5,
        filter_time=10,
        filter_z=1
    )

    try:
        merger = SlocumMerger(
            binary_path,
            ascii_path,
            cache_directory=config_path,
        )
        for p in merger.convert():
            args['file'] = p['ascii']
            create_dataset(**args)
    finally:
        # Cleanup
        shutil.rmtree(ascii_path, ignore_errors=True)  # Remove generated ASCII
        shutil.rmtree(netcdf_path, ignore_errors=True)  # Remove generated netCDF
        # Remove any cached .cac files
        for cac in glob(os.path.join(binary_path, '*.cac')):
            os.remove(cac)
