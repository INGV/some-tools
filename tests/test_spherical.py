from obspy import read
import numpy as np
from some_tools import spherical_conversion as SC


def rad2deg(rad):
    return 180. * rad / np.pi


def test_spherical():
    """ Check the correct functioning of the core module function"""

    ZZ = np.array([1., -3., 3., 1, 0, 1., -1])
    NN = np.array([0., 0., 0., 0., 0., 0., 0.])
    EE = np.array([2., 1., -1., 1., 1., 0., 0.])
    #
    check_rho = np.array([2.23606798,  3.16227766,  3.16227766,  1.41421356,
                          1.0, 1.0, 1.0])
    check_theta = np.array([0.46364761, -1.24904577,  1.24904577,  0.78539816,
                            0.0, 1.57079633, -1.57079633])
    check_phi = np.array([1.57079633,  1.57079633,  4.71238898,  1.57079633,
                          1.57079633, 0.0,  0.0])

    # Initialize fake object to check `_spherical_coord`
    _sc = SC.SphericalStream(read())
    rho, theta, phi = _sc._spherical_coords(ZZ, NN, EE)
    # print("Z\tN\tE\t\tRho\tTheta\tPhi")
    # for i in range(len(ZZ)):
    #     # convert to degree angle
    #     print("%.1f\t%.1f\t%.1f\t\t%.1f\t%.2f\t%.2f" % (
    #         ZZ[i], NN[i], EE[i], rho[i], rad2deg(theta[i]), rad2deg(phi[i])))

    np.testing.assert_array_almost_equal(rho, check_rho, decimal=3)
    np.testing.assert_array_almost_equal(theta, check_theta)
    np.testing.assert_array_almost_equal(phi, check_phi)
