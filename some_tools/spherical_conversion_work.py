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


class SphericalStream(object):
    def __init__(self, inst, convert=False):
        if not isinstance(inst, (Trace, Stream)):
            raise ValueError("I need an obspy.Stream object. Given type [%s]" %
                             type(inst))
        #
        self.inst = inst
        self.spherical = None
        if convert:
            self.convert_stream()

    def _spherical_coords(Z_trace, N_trace, E_trace):
        # modulus
        rho_trace = np.sqrt(Z_trace ** 2 + N_trace ** 2 + E_trace ** 2)
        # inclination
        theta_trace = np.arcsin(Z_trace/rho_trace)
        # azimuth
        phi_trace = np.arctan2(E_trace,N_trace)
        # make the range between 0 and 360 for phi
        phi_trace = np.where(phi_trace < 0, phi_trace + 2.0 * np.pi, phi_trace)
        return rho_trace, theta_trace, phi_trace

