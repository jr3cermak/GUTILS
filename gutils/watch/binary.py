#!python
# coding=utf-8
import os
import sys
import argparse
from pathlib import Path

from pyinotify import (
    IN_CLOSE_WRITE,
    IN_MOVED_TO,
    Notifier,
    NotifierError,
    ProcessEvent,
    WatchManager
)

from gutils import setup_cli_logger
from gutils.slocum import SlocumMerger

import logging
L = logging.getLogger(__name__)


class Binary2AsciiProcessor(ProcessEvent):

    def my_init(self, deployments_path, **kwargs):
        self.deployments_path = deployments_path

    def process_IN_CLOSE(self, event):
        self.convert_to_ascii(event)

    def process_IN_MOVED_TO(self, event):
        self.convert_to_ascii(event)


class Slocum2AsciiProcessor(Binary2AsciiProcessor):

    # (flight, science) file pairs
    PAIRS = {
        '.dbd': '.ebd',
        '.sbd': '.tbd',
        '.mbd': '.nbd'
    }

    def __call__(self, event):
        # Only fire events for the FLIGHT files. The science file will be searched for but we don't
        # want to fire events for both flight AND science files to due race conditions down
        # the chain
        if os.path.splitext(event.name)[-1] in self.PAIRS.keys():
            super().__call__(event)

    def check_for_pair(self, event):
        base_name, extension = os.path.splitext(event.name)

        # Look for the other file and append to the final_pair if it exists
        # If we got this far we already know the extension is in self.PAIRS.keys()
        oext = self.PAIRS[extension.lower()]
        possible_files = [
            os.path.join(event.path, base_name + oext),
            os.path.join(event.path, base_name + oext.upper())
        ]
        for p in possible_files:
            if os.path.isfile(p):
                _, file_ext = os.path.splitext(p)
                return [event.name, base_name + file_ext]

    def convert_to_ascii(self, event):
        file_pairs = self.check_for_pair(event)

        # Create a folder inside of the output directory for this glider folder name.
        # Assuming the binary file is in [rt|delayed]/binary, we just go back and add ascii
        binary_folder = Path(event.path)
        outputs_folder = binary_folder.parent / 'ascii'

        merger = SlocumMerger(
            event.path,
            outputs_folder,
            cache_directory=event.path,  # Default the cache directory to the data folder
            globs=file_pairs
        )
        merger.convert()


def create_ascii_arg_parser():

    parser = argparse.ArgumentParser(
        description="Monitor a directory for new binary glider data and outputs ASCII."
    )
    parser.add_argument(
        "-d",
        "--deployments_path",
        help="Path to deployments directory",
        default=os.environ.get('GUTILS_DEPLOYMENTS_DIRECTORY')
    )
    parser.add_argument(
        "-t",
        "--type",
        help="Glider type to interpret the data",
        default='slocum'
    )
    parser.add_argument(
        "--daemonize",
        help="To daemonize or not to daemonize",
        type=bool,
        default=False
    )

    return parser


def main_to_ascii():
    setup_cli_logger(logging.INFO)

    parser = create_ascii_arg_parser()
    args = parser.parse_args()

    if not args.deployments_path:
        L.error("Please provide a --deployments_path agrument or set the "
                "GUTILS_DEPLOYMENTS_DIRECTORY environmental variable")
        sys.exit(parser.print_usage())

    wm = WatchManager()
    mask = IN_MOVED_TO | IN_CLOSE_WRITE
    wm.add_watch(
        args.deployments_path,
        mask,
        rec=True,
        auto_add=True
    )

    # Convert binary data to ASCII
    if args.type == 'slocum':
        processor = Slocum2AsciiProcessor(
            deployments_path=args.deployments_path
        )
    notifier = Notifier(wm, processor, read_freq=10)  # Read every 10 seconds
    # Enable coalescing of events. This merges event types of the same type on the same file
    # together over the `read_freq` specified in the Notifier.
    notifier.coalesce_events()

    try:
        L.info(f"Watching {args.deployments_path} for new binary files")
        notifier.loop(daemonize=args.daemonize)
    except NotifierError:
        L.exception('Unable to start notifier loop')
        return 1

    L.info("GUTILS binary_to_ascii Exited Successfully")
    return 0
