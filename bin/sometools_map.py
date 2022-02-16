#!/usr/bin/env python

import logging
from obspy import read_events
from some_tools import plot_pygmt as PG

logging.basicConfig(level=logging.DEBUG)

cat = read_events()
SM = PG.SomeMapping()


SM.set_database(cat)
SM.set_gridfile("d")

import pdb; pdb.set_trace()
SM.plot_map()
