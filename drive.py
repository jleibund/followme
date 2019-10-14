"""
drive.py

Starts a driving loop

Usage:
    drive.py [--record]

Options:
  --record     record images to disk [default: False]
"""

import logging
import sys
import asyncio
from docopt import docopt
from rover import Rover

def setup_logging():
    '''
    Setup logging to output info to stdout
    '''
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    ch.setFormatter(formatter)
    root.addHandler(ch)


if __name__ == "__main__":
    arguments = docopt(__doc__)
    setup_logging()
    rover = Rover()
    rover.record = arguments["--record"]
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(rover.run())
    finally:
        loop.close()
