"""
Module containing utility functions and classes for converting seismograms
into spherical coordinates.

:copyright:
    INGV SOME project
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""

import copy
import numpy as np
from obspy import read, Trace, Stream
from some_tools import errors as STE
import logging

logger = logging.getLogger(__name__)


class SphericalStream(object):
    def __init__(self, inst, convert=False):
        if not isinstance(inst, (Trace, Stream)):
            raise ValueError("I need an obspy.Stream object. Given type [%s]" %
                             type(inst))
        #
        self.st = inst.copy()
        self.chan = ""
        self.stats = {}  # obspy trace stats dictionary
        self.out_st = Stream()
        #
        self.z_trace = Trace()
        self.e_trace = Trace()
        self.n_trace = Trace()
        #
        self.rho_trace = Trace()
        self.theta_trace = Trace()
        self.sin_theta_trace = Trace()
        self.sin_theta_pos_trace = Trace()
        self.phi_trace = Trace()
        self.sin_phi_trace = Trace()
        self.cos_phi_trace = Trace()
        #
        self._unpack_traces()
        self._check_channels()
        if convert:
            self.convert()

    def _unpack_traces(self):
        Z = self.st.select(channel="*Z")[0]
        if not Z:
            raise STE.MissingAttribute("Missing channel Z from input Stream!")

        E = self.st.select(channel="*E")[0]
        if not E:
            raise STE.MissingAttribute("Missing channel E from input Stream!")

        N = self.st.select(channel="*N")[0]
        if not N:
            raise STE.MissingAttribute("Missing channel N from input Stream!")
        #
        self.z_trace, self.e_trace, self.n_trace = Z, E, N

    def _check_channels(self):
        """ Using the Z component as reference for channel and stats"""
        self.chan = self.z_trace.stats.channel[:2]
        self.stats = self.z_trace.stats
        # Check if unique and warn me otherwise
        chlst = [tr.stats.channel[:2] for tr in self.st]
        _check = set(chlst)
        if len(_check) != 1:
            logger.warning(("Channel list not unique! %s! "
                            "Using Z channel for naming [%s]") % (
                            _check, self.chan))

    def _spherical_coords(self, Z, N, E):
        """ Given cartesian array, return Modulus (rho),
            Inclination (theta) and Azimuth (phi)
        """
        # --- Modulus
        _rho = np.sqrt(Z**2 + N**2 + E**2)

        # --- Inclination
        _theta = np.arcsin(Z/_rho)

        # --- Azimuth
        _phi = np.arctan2(E, N)
        # make the range between 0 and 360 for phi
        _phi = np.where(_phi < 0, _phi + 2.0 * np.pi, _phi)

        return _rho, _theta, _phi

    def _create_stream(self):
        """ Simply  translate and create spherical trace """

        Z = self.z_trace.data
        N = self.n_trace.data
        E = self.e_trace.data
        _rho, _theta, _phi = self._spherical_coords(Z, N, E)

        # --- RHO
        self.rho_trace.data = _rho
        self.rho_trace.stats = copy.deepcopy(self.stats)
        self.rho_trace.stats.channel = self.chan+"_RHO"

        # --- THETA
        self.theta_trace.data = _theta
        self.theta_trace.stats = copy.deepcopy(self.stats)
        self.theta_trace.stats.channel = self.chan+"_THE"

        self.sin_theta_trace.data = np.sin(_theta)
        self.sin_theta_trace.stats = copy.deepcopy(self.stats)
        self.sin_theta_trace.stats.channel = self.chan+"_SINTHE"

        self.sin_theta_pos_trace.data = np.abs(np.sin(_theta))
        self.sin_theta_pos_trace.stats = copy.deepcopy(self.stats)
        self.sin_theta_pos_trace.stats.channel = self.chan+"_SINTHEPOS"

        # --- PHI
        self.phi_trace.data = _phi
        self.phi_trace.stats = copy.deepcopy(self.stats)
        self.phi_trace.stats.channel = self.chan+"_PHI"

        self.sin_phi_trace.data = np.sin(_phi)
        self.sin_phi_trace.stats = copy.deepcopy(self.stats)
        self.sin_phi_trace.stats.channel = self.chan+"_SINPHI"

        self.cos_phi_trace.data = np.cos(_phi)
        self.cos_phi_trace.stats = copy.deepcopy(self.stats)
        self.cos_phi_trace.stats.channel = self.chan+"_COSPHI"

    def convert(self):
        """
        Creat a class-attribute stream with 7 components:
            rho, theta and phi (spherical coordinates) and
            sin(theta), sin(theta+), sin(phi) and cos(phi)
            sin(theta+) corresponds to the changing the sign whenever theta is negative
        """

        # Converts input stream and fill-in the 7-component trace
        logger.info("Creating spherical components ...")
        self._create_stream()

        self.out_st.id = ("Spherical coordinates: rho, theta, phi, sin(theta),"
                          " abs(sin(theta)), sin(phi), cos(phi)")
        self.out_st.append(self.rho_trace)
        self.out_st.append(self.theta_trace)
        self.out_st.append(self.phi_trace)
        self.out_st.append(self.sin_theta_trace)
        self.out_st.append(self.sin_theta_pos_trace)
        self.out_st.append(self.sin_phi_trace)
        self.out_st.append(self.cos_phi_trace)

    def get_converted_stream(self):
        """ Return the processed """
        if len(self.out_st) == 0:
            raise STE.MissingAttribute("Spherical stream is empty! "
                                       "Run `convert` method first.")
        #
        return self.out_st
