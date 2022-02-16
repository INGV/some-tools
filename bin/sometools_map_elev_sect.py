#!/usr/bin/env python

"""
This script is a wrapper around the `some_tools` plotting library
to create
 `sometools_map_elev_sect.py  DATABASE -g GRID -p PROFILE-COORD -y CONFIGURATION -s SAVEFIGNAME`

"""

import os
import sys
import argparse
import logging
from pathlib import Path
from pygmt import Figure
from some_tools import plot_pygmt as SPL

logging.basicConfig(format="%(name)s - %(funcName)s [%(levelname)s] %(message)s",
                    level=logging.INFO)

# ====================================================================
parser = argparse.ArgumentParser(
                description=(
                    "Script to create a complete figure of epicentral map " +
                    "plus an elevation (DEM) profile section and the " +
                    "relative depth section. It internally use the " +
                    "`sometools.plot_pygmt` " +
                    "module. Please refer to the project's book for a " +
                    "detailed reference"
                    ))
parser.add_argument('database', type=str,
                    help='path to database (*csv, pandas/geopandas DataFrame')
parser.add_argument('-g', '--gridfile', type=str, default=None,
                    help='path to a grid file (that PyGMT can read)')
parser.add_argument('-p', '--profile', type=float, nargs='+',
                    default=None,
                    help=('coordinates of the section_profile that will be' +
                          ' extracted. [lonStart, latStart, lonEnd, latEnd]'))
parser.add_argument('-y', '--configfile', type=str, default=None,
                    help=('YAML configuration file. If unspecified, default'+
                          ' will be used instead'))
parser.add_argument('-s', '--savefigname',  type=str, default=None,
                    help=('If specified, the figure will be saved with the'
                          'extension provided'))
# #
# parser.add_argument(
#                 '--printoutp', action='store_true', dest='printout_p',
#                 help="print the P-phases from INPUT CNV to OUTPUT CNV.")
# parser.add_argument(
#                 '--no-printoutp', action='store_false', dest='printout_p',
#                 help="DO NOT print the P-phases from INPUT CNV to OUTPUT CNV.")
# parser.add_argument('-v', '--version', action='version',
#                     version='%(prog)s ' + _VERSION)

# # ----------- Set Defaults
# parser.set_defaults(printout_p=True)
args = parser.parse_args()


# ingv_data = "Contains.20110101_20211231_Ml_2.8.Downloaded.20220216_1047.INGV.wildcsv"
# gapssdata = "gapss_initial_events.csv"
# demgrid = "/Users/matteo/ETH/UsefulDataset/LTGF.nc"

# =============== Instantiate the classes
SM = SPL.SomeMap(args.database)
SM.set_gridfile(args.gridfile)
SM.set_configfile(config_file=args.configfile)

SS = SPL.SomeSection(args.database)
SS.set_configfile(config_file=args.configfile)
if args.profile:
    SS.update_configuration(section_profile=args.profile)

SE = SPL.SomeElevation(args.gridfile)
SE.set_configfile(config_file=args.configfile)
if args.profile:
    SE.update_configuration(section_profile=args.profile)

if not args.configfile:
    # It means we are using the defaults, the default grid must be scaled!
    _region_sect1, _region_sect2, _, _ = SS.region
    _, _, _region_elev1, _region_elev2 = SE.region
    SE.region = [_region_sect1, _region_sect2, _region_elev1, _region_elev2]
    import pdb;  pdb.set_trace()


# =============== Plot
main_fig = Figure()

# Plot Map
main_fig = SM.plot_map(plot_config=args.configfile, show=False,
                       in_fig=main_fig, panel=None)

# Plot map-section
if args.profile:
    SECTION = args.profile
else:
    SECTION = SE.get_configdict()['section_profile']

main_fig.plot(x=[SECTION[0], SECTION[2]], y=[SECTION[1], SECTION[3]],
              pen="2p,red", straight_line=False)

# Plot Elevation
main_fig.shift_origin(yshift="-%f" % (SE.fig_height+1))  # yshift="h+0.1c")
main_fig = SE.plot_elevation_profile(plot_config=args.configfile,
                                     show=False, in_fig=main_fig, panel=None)

# Plot Section
main_fig.shift_origin(yshift="-%f" % SS.fig_height)
main_fig = SS.plot_section(plot_config=args.configfile, show=False,
                           in_fig=main_fig, panel=None)

# =============== Show + Save
main_fig.show("external")
if args.savefigname:
    print("Storing figure: %s" % args.savefigname)
    main_fig.savefig(args.savefigname)

print("DONE!")
