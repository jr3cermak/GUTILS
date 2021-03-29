#!/bin/bash
set -e
. /etc/profile

if [[ "$RUN_BINARY_TO_ASCII_WATCH" = "yes" ]]; then
    mkdir -p /etc/service/gutils_binary_to_ascii_watch
    cp $PROJECT_ROOT/docker/gutils_binary_to_ascii_watch /etc/service/gutils_binary_to_ascii_watch/run
fi

if  [[ "$RUN_ASCII_TO_NETCDF_WATCH" = "yes" ]]; then
    mkdir -p /etc/service/gutils_ascii_to_netcdf_watch
    cp $PROJECT_ROOT/docker/gutils_ascii_to_netcdf_watch /etc/service/gutils_ascii_to_netcdf_watch/run
fi

if  [[ "$RUN_NETCDF_TO_FTP_WATCH" = "yes" ]]; then
    mkdir -p /etc/service/gutils_netcdf_to_ftp_watch
    cp $PROJECT_ROOT/docker/gutils_netcdf_to_ftp_watch /etc/service/gutils_netcdf_to_ftp_watch/run
fi

if  [[ "$RUN_NETCDF_TO_ERDDAP_WATCH" = "yes" ]]; then
    mkdir -p /etc/service/gutils_netcdf_to_erddap_watch
    cp $PROJECT_ROOT/docker/gutils_netcdf_to_erddap_watch /etc/service/gutils_netcdf_to_erddap_watch/run
fi
