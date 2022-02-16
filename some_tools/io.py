"""
This module is in charge of loading and export all data types necessary
for SOME_TOOLS processing.

At the moment can load:
    - Pandas Dataframe
    - Standard CSV (with headers)
    - Obspy Catalog

"""

import yaml
from pathlib import Path
import logging
#
import pandas as pd
import obspy
#
import some_tools as ST
import some_tools.errors as STE


KM = 0.001
MT = 1000

logger = logging.getLogger(__name__)

# EVTID, EVTDATETIME, EVLA, EVLO, EVDP, EVMAG, EVMAGTYPE


# ==================================================================

def _get_conf(filepath, check_version=True):
    """ Simple function to unpack the YAML configuration file
        configuration file and return as a dict.
    """
    stver = ST.__version__

    # Create dict
    try:
        with open(filepath, "rt") as qfc:
            outDict = yaml.load(qfc, Loader=yaml.FullLoader)
    except KeyError:
        raise STE.BadConfigurationFile("Wrong key name/type!")

    # --- Check Versions
    if check_version:
        if stver.lower() != outDict['some_tools_version']:
            raise STE.BadConfigurationFile("SOME-TOOLS version [%s] and "
                                           "CONFIG version [%s] differs!" %
                                           (stver,
                                            outDict['some_tools_version']))
    #
    return outDict


# ==================================================================

class Loader(object):
    """ Main Class to import data. It will always return a custom
        formatted Pandas.DataFrame
    """
    def __init__(self, database):
        self.database = None
        self.df = None
        self._load_database(database)  # will import the database

    def _load_database(self, data):
        """ Switch among loading data ...
        """
        if isinstance(data, str):
            _data_path = Path(data)
            if _data_path.is_file() and _data_path.exists():
                _data_frame = pd.read_csv(_data_path)
                self._pandas2data(_data_frame)
            else:
                raise STE.FilesNotExisting("%s file do not exist!" % data)

        elif isinstance(data, pd.DataFrame):
            # pandas Dataframe --> simply append
            self._pandas2data(data)
            logger.debug("Recognized Pandas DataFrame data type ... loading")

        elif isinstance(data, obspy.Catalog):
            logger.debug("Recognized ObsPy Catalog data type ... loading")
            self._obspy2data(data)

        else:
            logger.error("Data type: %s  not yet supported! Contact maintaner" %
                         type(data))
        #
        logger.debug("Correctly loaded DATABASE!")

    def _pandas2data(self, data_frame):
        self.df = data_frame.copy()
        self._pd_select_columns()

    def _obspy2data(self, catalog, evid_prefix="opcat_"):
        """ If available, it will select the event's preferred solutions
        """
        evidlist = []
        utclist = []
        lonlist = []
        latlist = []
        deplist = []
        maglist = []
        magtype = []
        #
        for _xx, _ev in enumerate(catalog):
            if len(_ev.origins) != 0 and len(_ev.magnitudes) != 0:
                evidx = evid_prefix+str(_xx+1)
                # --- Select preferred solutions
                evor = _ev.preferred_origin()
                if not evor:
                    evor = _ev.origins[0]
                evmg = _ev.preferred_magnitude()
                if not evmg:
                    evmg = _ev.magnitudes[0]
                #
                evidlist.append(evidx)
                utclist.append(evor.time.datetime)
                lonlist.append(evor.longitude)
                latlist.append(evor.latitude)
                deplist.append(evor.depth*KM)
                maglist.append(evmg.mag)
                magtype.append(evmg.magnitude_type)
            else:
                logger.warning("(Event #%d) Have no origins OR magnitudes")

        # --- Create DATAFRAME
        self.df = pd.DataFrame(
                    {"ID": evidlist, "OT": utclist,
                     "LON": lonlist, "LAT": latlist, "DEP": deplist,
                     "MAG": maglist, "MAGTYPE": magtype}
                  )

    def _pd_add_evid_column(self):
        """ If you arrive here is because you need to create an
            automatic ID
        """
        self.df['ID'] = ["sometools_" + str(cc)
                         for cc in range(1, len(self.df)+1)]

    def _pd_select_columns(self):
        """ Select columns from pandas DataFrame and reorder them
            for the class API
        """
        _colnames = tuple(self.df.columns)
        _mandatory = set(("ID", "LON", "LAT", "DEP", "MAG"))
        # --- Search and change COLUMNS-NAME
        for cc in _colnames:
            if cc.lower().strip() in ("id", "event_id", "eqid", "eq_id", "evtid", "#"):
                self.df.rename(columns={cc: "ID"}, errors="raise", inplace=True)
            # Origin Time
            elif cc.lower().strip() in ("ot", "origin_time", "utc_datetime", "evtdatetime", "utc"):
                self.df.rename(columns={cc: "OT"}, errors="raise", inplace=True)
            # Longitude
            elif cc.lower().strip() in ("lon", "longitude", "ev_longitude", "evlo"):
                self.df.rename(columns={cc: "LON"}, errors="raise", inplace=True)
            # Latitude
            elif cc.lower().strip() in ("lat", "latitude", "ev_latitude", "evla"):
                self.df.rename(columns={cc: "LAT"}, errors="raise", inplace=True)
            # Depth
            elif cc.lower().strip() in ("dep", "depth", "ev_depth", "evdp"):
                self.df.rename(columns={cc: "DEP"}, errors="raise", inplace=True)
            # Magnitude
            elif cc.lower().strip() in ("mag", "magnitude", "ev_magnitude", "evmag"):
                self.df.rename(columns={cc: "MAG"}, errors="raise", inplace=True)
            # Magnitude Type
            elif cc.lower().strip() in ("magtype", "mag_type", "magnitude_type", "evmagtype"):
                self.df.rename(columns={cc: "MAGTYPE"}, errors="raise", inplace=True)
            else:
                continue

        # --- Extract
        _new_colnames = set(self.df.columns)
        _missing_fields = list(_mandatory.difference(_new_colnames))

        # Check if ID is missing and create a new one
        if "ID" in _missing_fields:
            self._pd_add_evid_column()
        # Recheck for other missing mandatory fields
        _new_colnames = set(self.df.columns)
        _missing_fields = list(_mandatory.difference(_new_colnames))
        if _missing_fields:
            raise STE.MissingParameter("I'm missing mandatory field: %s" %
                                       _missing_fields)

        # If arrived here, we have all mandatory fields !!!
        # Try to collect as much as possible, otherwise take only mandatory
        try:
            self.df = self.df[[
                "ID", "OT", "LON", "LAT", "DEP", "MAG", "MAGTYPE"]]
        except KeyError as err:
            logger.warning("Only MANDATORY field extracted. "
                           "Missing additional %s" %
                           err.args[0].split("not")[0].strip())
            self.df = self.df[_mandatory]

    def get_database(self, copy=False):
        if self.df is None or self.df.empty or not isinstance(self.df, pd.DataFrame):
            raise STE.MissingAttribute("Missing database!")
        if copy:
            return self.df .copy()
        else:
            return self.df

    def set_database(self, database):
        logger.warning("Overriding class database!")
        self._load_database(database)
