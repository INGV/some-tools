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
import obspy
import numpy as np
import pandas as pd
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


# ==========================================================


class MannekenPix2metadata(object):
    """

    Aquila-Dataset module

        possible attributes:
            stream_id --> identifier of resulting file

          |         |         |         |         |         |         |         |         |         |
0123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012
 FILE-NR YEAR MO DA HR MN  SEC     LAT      LON    DEPTH  MAG NETW STAT CMP EPIDIST SAM A P AUTOMAT W F
                                 DG.ddd   DG.ddd    KM                        KM     Hz   S   SEC
       3 2009  4 16  6 52 44.000 42.218N  13.109E  10.00 **** MN   AQU  ZNE   28.71 125 0 P ******* * *
       3 2009  4 16  6 52 44.000 42.218N  13.109E  10.00 **** MN   AQU  ZNE   28.71 125 0 S ******* * *
       6 2009  4 16  6 52 44.000 42.218N  13.109E  10.00 **** IV   CAMP ZNE   43.10 125 0 P ******* * *
       6 2009  4 16  6 52 44.000 42.218N  13.109E  10.00 **** IV   CAMP ZNE   43.10 125 0 S ******* * *
      12 2009  4 16  6 52 44.000 42.218N  13.109E  10.00 **** IV   CESI ZNE   89.05 125 1 P 179.115 1 D

    """
    def __init__(self, filepath, **kwargs):
        self.meta = {
            'NETWORK': [],
            'STATION': [],
            'PICKTIME': [],
            'PHASE': [],
            'WEIGHT': [],
            'POLARITY': [],
            'EPIDIST': [],
            'SAMPLING': [],
            'STATUS': []
        }
        self.pth = Path(filepath)

    def _unpack_line(self, inline):
        # out_dict = dict.fromkeys(self.columns_name)
        #
        # if inline[102] != "*":
        if inline[92:99] != "*******":
            # ---> WE HAVE A PHASE-PICK <---
            self.meta['NETWORK'].append(inline[62:64])
            self.meta['STATION'].append(inline[67:71].strip())
            _ref_utc = obspy.UTCDateTime(
                "%04d-%02d-%02dT%02d:%02d:%02.4f" % (
                    np.int(inline[9:13]),   # yr
                    np.int(inline[14:16]),  # mn
                    np.int(inline[17:19]),  # dd
                    np.int(inline[20:22]),  # hr
                    np.int(inline[23:25]),  # mn
                    np.float(inline[26:32]) # ss
                    )
                )
            _pick_utc = _ref_utc + np.float(inline[92:99])
            # self.meta['PICKTIME'].append("%s" % _pick_utc.datetime)
            self.meta['PICKTIME'].append(_pick_utc)
            self.meta['PHASE'].append(inline[90])
            self.meta['WEIGHT'].append(inline[100])
            #
            if inline[102] == "U":
                self.meta['POLARITY'].append("positive")
            elif inline[102] == "D":
                self.meta['POLARITY'].append("negative")
            else:
                self.meta['POLARITY'].append("undecidable")
            #
            self.meta['EPIDIST'].append(np.float(inline[76:83]))
            self.meta['SAMPLING'].append(np.float(inline[84:87]))
            #
            if inline[88] == "1":
                self.meta['STATUS'].append("automatic")
            else:
                self.meta['STATUS'].append("manual")

    def read_file(self):
        # picked_stations = []
        # metadata = {}
        with self.pth.open(mode='r') as IN:
            for xx, line in enumerate(IN):
                if xx >= 16:
                    if not line.strip():
                        break
                    else:
                        # collect
                        self._unpack_line(line)

    def extract_stream(self, dir_path):
        """ Extract station stream from dir """
        st = obspy.core.Stream()
        for stat in self.meta["STATION"]:
            st += obspy.read(dir_path+"/*"+stat.lower()+"*")
        #
        return st

    def set_file(self, filepath):
        if isinstance(filepath, str) and Path(filepath).isfile:
            self.pth = Path(filepath)
            logger.info("Resetting class metadata!")
            self._read_file()
        else:
            raise ValueError("%r is not a file!" % filepath)

    def get_metadata(self):
        """ Return a data frame object from the stored dir """
        _df = pd.DataFrame.from_dict(self.meta)
        _df.sort_values(by="EPIDIST", inplace=True, ignore_index=True)
        return _df

    def store_metadata(self, outfile, floatformat="%.3f"):
        _df = pd.DataFrame.from_dict(self.meta)
        _df.sort_values(by="EPIDIST", inplace=True, ignore_index=True)
        _df.to_csv(outfile,
                   sep=',',
                   index=False,
                   float_format=floatformat,
                   na_rep="NA", encoding='utf-8')

    def get_picked_stations(self):
        """ return the list of picked station """
        return tuple(self.meta["STATION"])


class AquilaDS2seisbench(object):
    """
    This class will unpack the final dataset

    """

    def __init__(self, work_dir):
        self.seisbench_columns = (
            # Trace
            'trace_start_time',
            'trace_dt_s',
            'trace_npts',
            'trace_name',
            'trace_polarity'  # relative to P
            'trace_eval_p',                 'trace_eval_s',
            'trace_p_status',               'trace_s_status',
            'trace_p_weight',               'trace_s_weight',
            'trace_p_arrival_time',         'trace_s_arrival_time',
            'trace_p_arrival_sample',       'trace_s_arrival_sample',
            'trace_p_uncertainty_s',        'trace_s_uncertainty_s',
            # Station
            'station_network_code',
            'station_code',
            'station_location_code',
            'station_channel',
            'station_latitude_deg',
            'station_longitude_deg',
            'station_elevation_m',
            # Path
            'path_p_travel_s',              'path_s_travel_s',
            'path_p_residual_s',            'path_s_residual_s',
            'path_weight_phase_location_p', 'path_weight_phase_location_s',
            'path_azimuth_deg',
            'path_back_azimuth_deg',
            'path_ep_distance_km',
            'path_hyp_distance_km',
            # Source
            'source_type'
            'source_origin_time',
            'source_latitude_deg',
            'source_longitude_deg',
            'source_depth_km',
            'source_origin_uncertainty_s',
            'source_latitude_uncertainty_deg',
            'source_longitude_uncertainty_deg',
            'source_depth_uncertainty_km',
            'source_stderror_s',
            'source_gap_deg',
            'source_horizontal_uncertainty_km',
            'source_magnitude',
            'source_magnitude_type'
        )
        self.meta = None  # will be a Pandas Dataframe
        self.st = None  # obspy stream containing traces
        #
        if not isinstance(work_dir, Path):
            self.work_dir = Path(work_dir)
        else:
            self.work_dir = work_dir
        #
        self.phase_file = self.work_dir / "AquilaPSRun2.Sum"
        self.event_file = self.work_dir / "Run2_reloc2.sum"
        if not self.phase_file.exists() or not self.event_file.exists():
            raise AttributeError("Missing PHASEFILE or EVENTFILE in working dir!")
        #
        self.df = pd.DataFrame()

    def _fill_na_fields(self, indict):
        """ Fill empty fields with None """
        _tmp_ref = set(self.seisbench_columns)
        _tmp = set(indict)
        #
        _miss_kk = tuple(_tmp_ref - _tmp)
        if _miss_kk:
            up_dict = {key: None for key in _miss_kk}
            return indict.update(up_dict)
        else:
            return indict

    def _sanity_check_df(self, indf):
        """ Check that no missing keys / additional keys are inserted """
        _tmp_ref = set(self.seisbench_columns)
        _tmp = set(indf.columns())
        #
        _miss_kk = tuple(_tmp_ref - _tmp)
        _add_kk = tuple(_tmp - _tmp_ref)
        #
        if _miss_kk:
            raise ValueError("Missing keys:  %r" % _miss_kk)
        elif _add_kk:
            raise ValueError("Additional keys:  %r" % _add_kk)
        else:
            # Columns matches reference
            return True

    # -----------------------------------------------------
    def _mannekenPix2metadata(self):
        """ Extract MPX metadata and store to working dict.

        Example of extraction MPX object

      NETWORK STATION                    PICKTIME PHASE WEIGHT  POLARITY
    0      IV    RM21  2009-11-29 21:37:55.424000     P      3  positive
    EPIDIST  SAMPLING     STATUS
       6.90     125.0  automatic

        This method will fill in the following fields:

            - 'station_network_code'
            - 'station_code'
            - 'trace_p_arrival_time' / 'trace_s_arrival_time'
            - 'path_ep_distance_km'
            - 'trace_p_weight' / 'trace_s_weight'
            - 'trace_polarity_p' / 'trace_polarity_p'
            - 'trace_p_status', 'trace_s_status', # automanu

        """
        _mpx = MannekenPix2metadata(self.phase_file)
        _mpx.read_file()
        _tmp_df = _mpx.get_metadata()

        # Map dictionary columns
        out_df = pd.DataFrame(columns=[
             'station_network_code',
             'station_code',
             'trace_p_arrival_time', 'trace_s_arrival_time',
             'path_ep_distance_km',
             'trace_p_weight', 'trace_s_weight',
             'trace_polarity_p', 'trace_polarity_s', 'trace_polarity',
             'trace_p_status', 'trace_s_status'
            ])

        for index, row in _tmp_df.iterrows():
            dd = {}
            dd['station_network_code'] = row["NETWORK"]
            dd['station_code'] = row["STATION"]
            dd['path_ep_distance_km'] = row["EPIDIST"]
            #
            if row["PHASE"] == "P":
                dd['trace_p_arrival_time'] = row["PICKTIME"]
                dd['trace_p_weight'] = row["WEIGHT"]
                dd['trace_polarity_p'] = row["POLARITY"]
                dd['trace_p_status'] = row["STATUS"]
            elif row["PHASE"] == "S":
                dd['trace_s_arrival_time'] = row["PICKTIME"]
                dd['trace_s_weight'] = row["WEIGHT"]
                dd['trace_polarity_s'] = row["POLARITY"]
                dd['trace_s_status'] = row["STATUS"]
            else:
                raise ValueError("UNKNOWN PHASE:  %r  !" % row["PHASE"])
            #
            out_df = out_df.append(dd, ignore_index=True)

        out_df["UNIQ"] = out_df["station_network_code"] + "." + out_df["station_code"]
        out_df = out_df.groupby("UNIQ", as_index=False).agg('first')
        out_df['trace_polarity'] = out_df['trace_polarity_p']
        out_df.sort_values("path_ep_distance_km", ignore_index=True, inplace=True)

        return out_df, out_df['station_code']

    def _extract_stream(self, picked_stations):
        """ Extract station stream from dir """
        st = obspy.core.Stream()
        for stat in picked_stations:
            st += obspy.read(str(self.work_dir) + "/*"+stat.lower()+"*")
        #
        return st

    def _extract_stream_meta(self, indict):
        """ Extract metadata from stream attribute

        'stla': 42.627998, 'stlo': 13.142, 'stel': 1.454,
        'evla': 42.585999, 'evlo': 13.2177, 'evdp': 5350.0,
        'dist': 7.7688251, 'az': 306.93469, 'baz': 126.88363,

        To be faster we populate a dictionary and then transform it into
        a pandas.DataFrame

        Extract indirectly (aftermerging):
            - path_travel_time_P_s
            - path_travel_time_S_s
            - path_residual_P_s
            - path_residual_S_s
            - path_ep_distance_km
            - path_hyp_distance_km

        """

        if not self.st:
            raise AttributeError("Missing Stream! Run `_extract_stream` first!")

        # In dict contains the MPX phases metadata. We need here to
        # complement those by adding the metadata from stream-trace of the
        # same network-stationcode pairs.
        # The traces refers all to the picked traces

        _dd = {
            # Mandatory
            'station_network_code': [],
            'station_code': [],
            # Station
            'station_location_code': [],
            'station_channel': [],
            'station_latitude_deg': [],
            'station_longitude_deg': [],
            'station_elevation_m': [],
            # Source
            'source_id': [],
            'source_type': [],
            # 'source_origin_time': [],
            'source_latitude_deg': [],
            'source_longitude_deg': [],
            'source_depth_km': [],
            'source_gap_deg': [],
            # Path
            'path_azimuth_deg': [],
            'path_back_azimuth_deg': [],
            # Trace
            'trace_name': [],
            'trace_start_time': [],
            'trace_dt_s': [],
            'trace_npts': [],
            'trace_z_min_counts': [],
            'trace_n_min_counts': [],
            'trace_e_min_counts': [],
            'trace_z_max_counts': [],
            'trace_n_max_counts': [],
            'trace_e_max_counts': [],
            'trace_z_mean_counts': [],
            'trace_n_mean_counts': [],
            'trace_e_mean_counts': [],
            'trace_z_median_counts': [],
            'trace_n_median_counts': [],
            'trace_e_median_counts': [],
            'trace_z_rms_counts': [],
            'trace_n_rms_counts': [],
            'trace_e_rms_counts': [],
            'trace_z_lower_quartile_counts': [],
            'trace_n_lower_quartile_counts': [],
            'trace_e_lower_quartile_counts': [],
            'trace_z_upper_quartile_counts': [],
            'trace_n_upper_quartile_counts': [],
            'trace_e_upper_quartile_counts': [],
            }

        for tr in self.st:
            # To merge with previous dataframe
            _dd['station_network_code'].append(tr.stats.network)
            _dd['station_code'].append(tr.stats.station)

            # Station
            _dd['station_channel'].append(tr.stats.sac.kevnm[-3:-1])
            _dd['station_location_code'].append('')  # To ask Carlo
            _dd['station_latitude_deg'].append(tr.stats.sac.stla)
            _dd['station_longitude_deg'].append(tr.stats.sac.stlo)
            _dd['station_elevation_m'].append(tr.stats.sac.stel * MT)

            # Source
            _dd['source_id'].append("AQ"+tr.stats.sac.kevnm[:-3])
            _dd['source_type'].append('earthquake')
            _dd['source_latitude_deg'].append(tr.stats.sac.evla)
            _dd['source_longitude_deg'].append(tr.stats.sac.evlo)
            _dd['source_depth_km'].append(tr.stats.sac.evdp)
            _dd['source_gap_deg'].append(tr.stats.sac.evdp)

            # Path
            _dd['path_azimuth_deg'].append(tr.stats.sac.az)
            _dd['path_back_azimuth_deg'].append(tr.stats.sac.baz)

            # Trace
            _dd['trace_name'].append('.'.join([
                                "AQ"+tr.stats.sac.kevnm[:-3],
                                tr.stats.network,
                                tr.stats.station,
                                "", tr.stats.sac.kevnm[-3:-1]]))
            # _dd['trace_start_time'].append("%s" % tr.stats.starttime.datetime)
            _dd['trace_start_time'].append(tr.stats.starttime)
            _dd['trace_dt_s'].append(tr.stats.delta)
            _dd['trace_npts'].append(tr.stats.sac.npts)

            for _comp in ("n", "e", "z"):

                if _comp == tr.stats.sac.kcmpnm.lower():
                    _dd['trace_%s_min_counts' % _comp].append(
                        np.min(tr.data)
                        )
                    _dd['trace_%s_max_counts' % _comp].append(
                        np.max(tr.data)
                        )
                    _dd['trace_%s_mean_counts' % _comp].append(
                        np.mean(tr.data)
                        )
                    _dd['trace_%s_median_counts' % _comp].append(
                        np.median(tr.data)
                        )
                    _dd['trace_%s_rms_counts' % _comp].append(
                        np.sqrt(np.mean(np.square(tr.data)))
                        )
                    _dd['trace_%s_lower_quartile_counts' % _comp].append(
                        np.quantile(tr.data, 0.25)
                        )
                    _dd['trace_%s_upper_quartile_counts' % _comp].append(
                        np.quantile(tr.data, 0.75)
                                    )
                else:
                    _dd['trace_%s_min_counts' % _comp].append(np.nan)
                    _dd['trace_%s_max_counts' % _comp].append(np.nan)
                    _dd['trace_%s_mean_counts' % _comp].append(np.nan)
                    _dd['trace_%s_median_counts' % _comp].append(np.nan)
                    _dd['trace_%s_rms_counts' % _comp].append(np.nan)
                    _dd['trace_%s_lower_quartile_counts' % _comp].append(np.nan)
                    _dd['trace_%s_upper_quartile_counts' % _comp].append(np.nan)

        # Finish loading traces:
        _df = pd.DataFrame.from_dict(_dd)
        _df["UNIQ"] = _df["station_network_code"] + "." + _df["station_code"]
        _df = _df.groupby("UNIQ", as_index=False).agg('first')
        # Merge MPX on UNIQ
        _df_all = _df.merge(indict, how='outer', on="UNIQ")

        # Calculate Additional Feature
        self._add_features(_df_all)

        # Sort and allocate to self attribute
        _df_all.sort_values("path_ep_distance_km", ignore_index=True, inplace=True)
        return _df_all

    def _add_features(self, indf):
        """ Append the following columns:

            - path_p_travel_s  trace_p_arrival_sample
            - path_s_travel_s  trace_s_arrival_sample
            - path_hyp_distance_km

        """
        def __convert_utc_to_string(row):
            row["trace_start_time"] = ("%s" % row["trace_start_time"])
            row["trace_p_arrival_time"] = ("%s" % row["trace_p_arrival_time"]).datetime
            row["trace_s_arrival_time"] = ("%s" % row["trace_s_arrival_time"]).datetime
        # ====

        indf["path_hyp_distance_km"] = (
            np.sqrt(indf['path_ep_distance_km']**2 +
                    (indf['source_depth_km'] + indf["station_elevation_m"])**2
                    ))
        #
        indf['path_p_travel_s'] = (indf['trace_p_arrival_time'] -
                                   indf['trace_start_time'])
        indf['trace_p_arrival_sample'] = (indf['path_p_travel_s'] /
                                          indf['trace_dt_s'])
        #
        indf['path_s_travel_s'] = (indf['trace_s_arrival_time'] -
                                   indf['trace_start_time'])
        indf['trace_s_arrival_sample'] = (indf['path_s_travel_s'] /
                                          indf['trace_dt_s'])
        return indf

    def orchestrator(self):
        """ This method takes care of extracting everything
            and orchestrate """

        # Extract phase pick information from MPX.Sum
        _mpx_df, picked_stations = self._mannekenPix2metadata()

        # Import picked waveforms and compute remaining metadata
        # NB!! They must be in the same directory as MPX.Sum files!
        self.st = self._extract_stream(picked_stations)

        # Extract stream meta
        _meta_df = self._extract_stream_meta(_mpx_df)

        # Allocate
        self.meta = _meta_df.copy(deep=True)

    def set_file(self, filepath):
        if isinstance(filepath, str) and Path(filepath).isfile:
            self.pth = Path(filepath)
            logger.info("Resetting class metadata!")
            self._read_file()
        else:
            raise ValueError("%r is not a file!" % filepath)

    # def get_metadata(self):
    #     """ Return a data frame object from the stored dir """
    #     _df = pd.DataFrame.from_dict(self.meta)
    #     self.meta.sort_values(by="EPIDIST", inplace=True, ignore_index=True)

    def store_metadata(self, outfile, floatformat="%.3f"):
        self.meta.to_csv(
                    outfile,
                    sep=',',
                    index=False,
                    float_format=floatformat,
                    na_rep="NA", encoding='utf-8')


"""

INGV                            SEISBENCH
path_travel_time_P_s            path_p_travel_s
path_travel_time_S_s            path_s_travel_s
path_residual_P_s               path_p_residual_s
path_residual_S_s               path_s_residual_s
path_ep_distance_km
path_hyp_distance_km
path_azimuth_deg
path_backazimuth_deg            path_back_azimuth_deg
path_weight_phase_location_P
path_weight_phase_location_S

trace_start_time
trace_dt_s
trace_npts
trace_polarity
trace_eval_P
trace_P_uncertainty_s
trace_P_arrival_time
trace_P_arrival_sample
trace_eval_S
trace_S_uncertainty_s
trace_S_arrival_time
trace_S_arrival_sample



station_network_code  station_code  station_location_code  station_channels


"""
