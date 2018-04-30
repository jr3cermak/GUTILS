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
    'bass-20160909T1733',
])
def test_real_deployments(deployment):
    setup_testing_logger(level=logging.WARNING)
    binary_path = resource('slocum', 'binary', deployment)
    ascii_path = resource('slocum', 'ascii', deployment)
    netcdf_path = resource('slocum', 'netcdf', deployment)
    default_configs = resource('slocum', 'config', deployment)

    # Config path is usually an env variable pointing to a configuration setup
    all_config_path = os.environ.get('GUTILS_TEST_CONFIG_DIRECTORY', default_configs)
    config_path = os.path.join(all_config_path, deployment)

    # Static args
    args = dict(
        reader_class=SlocumReader,
        config_path=config_path,
        output_path=netcdf_path,
        subset=True,
        template='ioos_ngdac',
        profile_id_type=2,
        filter_distance=1,
        filter_points=5,
        filter_time=10,
        filter_z=1
    )

    try:
        merger = SlocumMerger(
            binary_path,
            ascii_path
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
