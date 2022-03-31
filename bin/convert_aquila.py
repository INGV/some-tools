#!/usr/bin/env python

""" The folder /data/AQUILA/2009-Aquila is structured as follows:
    - DAY
        - EVENT
            - EVENTID stations (statname and component. all lowercase)
            - AquilaPSRun2.Phs (phasefile in custom format)
        *** - AquilaPSRun2.Sum (containing the PhasePicks in MPX format)
            - Run2_reloc2.arc  (reloc+phases in HYPOINVERSE format)
        *** - Run2_reloc2.sum  (reloc wildcard from HYPOINVERSE)

    The *** are used for metadata collection.

    SEISBENCH format:
    https://seisbench.readthedocs.io/en/stable/pages/data_format.html#source-parameters

"""

import os
import sys
import argparse
from pathlib import Path
import datetime
import multiprocessing as mp
from tqdm import tqdm
from time import sleep
from some_tools import io
from obspy import UTCDateTime
import concurrent
import itertools


# =================== Parsing input
parser = argparse.ArgumentParser(
            description=("Script to convert AQUILA ob pickle outputs into "
                         "METADATA-CSV and H5DF storage files for SEISBENCH "
                         " and ML use."))
parser.add_argument('-d', '--datapath', type=str,
                    help='path to AQUILA dataset structure. MAIN folder!'
                         'Usage of wildcard is allowed.')
parser.add_argument('-c', '--cpulen', type=str,  # nargs='+',
                    help='number of CPUs to be used for conversion. '
                         'If higher than the total available, '
                         'all of them will be used')
parser.add_argument('-m', '--metadataoutdir', type=str,
                    help='path of the output EVENT metadata csv.')
parser.add_argument('-f', '--outputh5df', type=str,
                    help='path of the H5DF output (containing traces).')
parser.set_defaults(cpuslen=10)
parser.set_defaults(metadataoutdir=Path.home() / "Aquila_DB_Conversion")


args = parser.parse_args()

# =================== Parsing
if len(sys.argv) == 1:
    sys.stderr.write(("USAGE: %s --help"+os.linesep) % (
                        Path(sys.argv[0]).name
                    ))
    sys.exit()

elif not args.datapath or not args.metadataoutdir:
    sys.stderr.write("I need an input and output Path! [-d ; -m]" +
                     os.linesep)
    sys.exit()


# =================== Functions

def process_day_mp(mparg):
    """ We extract SUM files """

    daypath, storepath = mparg
    missing_events = []
    eventdir = [e for e in daypath.iterdir() if e.is_dir()]
    # print("Working with DAY: %s" % daypath.name)
    for xx, ev in enumerate(tqdm(eventdir,
                            bar_format='{l_bar}{bar:25}{r_bar}{bar:-25b}')):
        eqid = ev.name
        #
        try:
            AQ = io.AquilaDS2seisbench(ev)
        except AttributeError:
            missing_events.append(eqid)
            continue
        #

        AQ.orchestrator()
        # MPX, MDF = AQ._mannekenPix2metadata()
        # MPX.store_metadata(storepath / ("%s.metadata.EVENT.csv" % eqid))
        # import pdb; pdb.set_trace()


if len(sys.argv) == 1:
    sys.stderr.write(("USAGE: %s --help"+os.linesep) % (
                        Path(sys.argv[0]).name
                    ))
    sys.exit()

elif not args.datapath:
    sys.stderr.write("Let me know where to search first ... I need a Path! [-p]" +
                     os.linesep)
    sys.exit()


# ======================================  MAIN
# ==============================================

rootpath = Path(args.datapath)
outpath = Path(args.metadataoutdir)

print()
print(os.linesep + "INPUT DIR:  %s" % rootpath)
print(("STORE DIR:  %s" + os.linesep*2) % outpath)

# Prepare output dir
if not outpath.exists() or not outpath.is_dir():
    outpath.mkdir(parents=True, exist_ok=True)
elif outpath.exists() and outpath.is_dir():
    if any(outpath.iterdir()):
        # Directory NOT empty
        eraseme = input(
            ("Directory not empty: %s " + os.linesep + " ... overriding? [y]: ")
            % outpath)
        if not eraseme or str(eraseme).lower() in ("yes", "y"):
            print("ERASING ..." + os.linesep)
            import shutil
            shutil.rmtree(outpath)
            outpath.mkdir(parents=True, exist_ok=True)
        else:
            print("Nothing happened!" + os.linesep)

# -------------------------------------------------
startt = UTCDateTime()
print("Start:  %s" % startt)

daydir = [x for x in rootpath.iterdir() if x.is_dir()]
daydir = daydir[0:10]
process_day_mp((daydir[0], outpath))




# ======== PRODUCTION
# functarg = [tuple(c) for c in zip(daydir, itertools.repeat(outpath))]
# with concurrent.futures.ProcessPoolExecutor() as executor:
#     results = executor.map(process_day_mp, functarg)
#     # --- Do something with out
#     # for ii in results:
#     #     print(any(ii))

endt = UTCDateTime()
print("End:  %s" % UTCDateTime())
print("Elapsed time (hr) --->  %.2f" % ((endt - startt) / 3600.0))
print("Elapsed time (sec) --->  %.2f" % (endt - startt))


"""
https://stackoverflow.com/questions/59095085/processpoolexecutor-pass-multiple-arguments

Savior:

With multiprocessing Pool you would use starmap() and it would use start * to
unpack tuple to arguments

ExampleFunct( *('277906', 'cA2i150s81HI3qbq1fzi', 'za1Oq5CGHj3pkkXWNghG') )
ExampleFunct( *('213674', 'cA2i150s81HI3qbq1fzi', 'za1Oq5CGHj3pkkXWNghG') )

It seems concurrent.futures.ProcessPoolExecutor doesn't have starmap()
so it sends it as one argument - tuple

ExampleFunct( ('277906', 'cA2i150s81HI3qbq1fzi', 'za1Oq5CGHj3pkkXWNghG') )
ExampleFunct( ('213674', 'cA2i150s81HI3qbq1fzi', 'za1Oq5CGHj3pkkXWNghG') )

and you would need to unpack it inside function

def __init__(self, data):
    player_id, match_id, match_id_team = data
"""
