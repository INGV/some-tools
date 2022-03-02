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


def spherical_coords(Z_trace, N_trace, E_trace):
    # modulus
    rho_trace = np.sqrt(Z_trace ** 2 + N_trace ** 2 + E_trace ** 2)
    # inclination
    theta_trace = np.arcsin(Z_trace/rho_trace)
    # azimuth
    phi_trace = np.arctan2(E_trace,N_trace)
    # make the range between 0 and 360 for phi
    phi_trace = np.where(phi_trace < 0, phi_trace + 2.0 * np.pi, phi_trace)
    return rho_trace, theta_trace, phi_trace


def build_sphe_stream(st):
    """
    Input a stream consisting of N,E,Z recorded traces that must have been 'detrended' and 'demeaned'
    Requires  Trace and Stream from obspy and copy

    Returns a stream with 7 components:
        rho, theta and phi (spherical coordinates) and
        sin(theta), sin(theta+), sin(phi) and cos(phi)
        sin(theta+) corresponds to the changing the sign whenever theta is negative
    """
    # store the input data
    for i in range(len(st)):
        comp = st[i].stats.channel[-1]
        if comp == 'E':
            E = st[i].data
        elif comp == 'N':
            N = st[i].data
        elif comp == 'Z':
            Z = st[i].data
        else:
            print ("compoment not recognized")
            quit()
    #
    # rxtract the sampling and the gain of the channle
    samp_gain_cha = st[0].stats.channel[:2]
    # apply spherical coordinates transformation
    rho, theta, phi = spherical_coords(Z,N,E)
    #
    # comment to be added into the processing metadata of the traces
    processing = ["spherical coordinates: rho, theta, phi, sin(theta), abs(sin(theta)), sin(phi), cos(phi)"]

    # initialize list to contain the channels
    sphe_traces = []

    # construct obspy.Trace objects for each component
    # --- Modulus ($\rho$)

    rho_trace = Trace()
    rho_trace.data = rho
    sphe_traces.append(rho_trace)

    # --- Inclination Theta ($\theta$)
    theta_trace = Trace()
    sin_theta_trace = Trace()
    sin_theta_pos_trace = Trace()
    #
    theta_trace.data = theta
    sin_theta_trace.data = np.sin(theta)
    #
    sin_theta_pos_trace.data = np.sin(theta)
    sin_theta_pos_trace.data = np.abs(sin_theta_pos_trace.data)

    sphe_traces.append(theta_trace)
    sphe_traces.append(sin_theta_trace)
    sphe_traces.append(sin_theta_pos_trace)

    # --- Azimuth Phi ($\phi$)
    phi_trace = Trace()
    sin_phi_trace = Trace()
    cos_phi_trace = Trace()
    #
    phi_trace.data = phi
    sin_phi_trace.data = np.sin(phi)
    cos_phi_trace.data = np.cos(phi)

    sphe_traces.append(phi_trace)
    sphe_traces.append(sin_phi_trace)
    sphe_traces.append(cos_phi_trace)

    # set the channel names.
    tmp = ["_RHO", "_THE", "_SINTHE", "_SINTHEPOS", "_PHI", "_SINPHI", "_COSPHI"]
    sphe_chans = []
    for c in tmp:
        sphe_chans.append(samp_gain_cha + c)

    # deep copy the metadata in stats and update them
    for i, comp in enumerate(sphe_traces):
        sphe_traces[i].stats = copy.deepcopy(st[0].stats)
        sphe_traces[i].stats.processing.append(processing)
        #print (i, comp)
        sphe_traces[i].stats.channel = sphe_chans[i]

    # initialize the spherical coordinates stream. Note the ordering!
    sphe_stream = Stream(traces=[rho_trace, theta_trace, phi_trace, sin_theta_trace, sin_theta_pos_trace, sin_phi_trace, cos_phi_trace])
    return sphe_stream


def rad2deg(rad):
    return 180. * rad / np.pi


if __name__ == "__main__":
    ZZ = np.array([1., -3., 3., 1, 0, 1., -1])
    NN = np.array([0., 0., 0., 0., 0., 0., 0.])
    EE = np.array([2., 1., -1., 1., 1., 0., 0.])
    #
    print("EXAMPLE")
    print("Z, N, E coordinates")
    print(ZZ)
    print(NN)
    print(EE)
    #
    rho, theta, phi = spherical_coords(ZZ,NN,EE)
    print("Z\tN\tE\t\tRho\tTheta\tPhi")
    for i in range(len(ZZ)):
        # convert to degree angle
        print("%.1f\t%.1f\t%.1f\t\t%.1f\t%.2f\t%.2f" % (ZZ[i], NN[i], EE[i], rho[i], rad2deg(theta[i]), rad2deg(phi[i])))

    st = read()
    # filtering
    st.detrend('linear')
    st = st.filter('bandpass', freqmin=0.2, freqmax=20.0)

    st.detrend('linear')
    st.detrend('constant')

    sphe_st = build_sphe_stream(st)

    sphe_st[:3].plot(outfile="spherical_rho_theta_phi.png", equal_scale=False)
    sphe_st[3:].plot(outfile="spherical_sin_cos_theta_phi.png", equal_scale=False)

    st_all = st + sphe_st[:3]
    st_all.plot(outfile="cart_sphe.png",equal_scale=False)
    sphe_st[3:].plot(outfile="spherical_sin_cos_theta_phi.png",equal_scale=False)

    st.plot(outfile="cartesian.png",equal_scale=False)


    print(st)
    print(sphe_st)

    #st.plot()
