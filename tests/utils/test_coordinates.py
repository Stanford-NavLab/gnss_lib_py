"""Tests for coordinate transformations.

"""

__authors__ = "Ashwin Kanhere"
__date__ = "21 Jun 2022"

import numpy as np
import pytest

import gnss_lib_py.utils.constants as consts
from gnss_lib_py.utils.coordinates import ecef_to_el_az
from gnss_lib_py.utils.coordinates import geodetic_to_ecef, ecef_to_geodetic, LocalCoord

@pytest.fixture(name="local_ecef")
def fixture_local_ecef():
    """Return ECEF origin for local NED frame of reference

    Returns
    -------
    local_ecef : np.ndarray
        3x1 array containing ECEF coordinates of NED origin [m , m, m]
    """
    local_ecef = np.reshape([-2700628.97971166, -4292443.61165747, 3855152.80233124], [3, 1])
    return local_ecef


@pytest.fixture(name="local_lla")
def fixture_local_lla():
    """Return LLA origin for local NED frame of reference

    Returns
    -------
    local_lla : np.ndarray
        3x1 array containing LLA coordinates of NED origin [deg, deg, m]
    """
    local_lla = np.reshape([37.427112, -122.1764146, 16], [3, 1])
    return local_lla


@pytest.fixture(name="local_frame")
def fixture_local_reference(local_lla):
    """Return NED local frame of reference

    Parameters
    ----------
    local_lla : np.ndarray
        LLA coordinates of NED origin 3x1 [deg, deg, m]

    Returns
    -------
    local_frame : gnss_lib_py.utils.coordinates.LocalCoord
        NED local frame of reference with given LLA as origin
    """
    local_frame = LocalCoord.from_geodetic(local_lla)
    return local_frame

@pytest.mark.parametrize("lla, exp_ecef",
                        [(np.array([[37.427112], [-122.1764146], [16]]),
                          np.array([[-2700628], [-4292443], [3855152]])),
                         (np.array([[51.4934], [0], [0]]),
                          np.array([[3979223], [0], [4967905]])),
                        (np.array([[0], [0], [0]]),
                         np.array([[6378137],[0], [0]])),
                        (np.array([[0], [-122.1764146], [0]]),
                         np.array([[-3396535], [-5398534], [0]]))
                        ])
def test_geodetic_to_ecef(lla, exp_ecef):
    """Test LLA (in WGS-84 reference ellipsoid) to ECEF conversion

    Parameters
    ----------
    lla : np.ndarray
        LLA test coordinates, shape: 3x1, units: [deg, deg, m]
    exp_ecef : np.ndarray
        Expected ECEF values (obtained using MATLAB's equivalent function)
    """
    ecef_orig = geodetic_to_ecef(lla)
    ecef_trans = geodetic_to_ecef(lla.T)
    lla_back = ecef_to_geodetic(ecef_orig)
    np.testing.assert_array_almost_equal(ecef_orig, exp_ecef, decimal=0)
    np.testing.assert_array_almost_equal(ecef_orig, ecef_trans.T)
    np.testing.assert_array_almost_equal(lla, lla_back, decimal=2)

@pytest.mark.parametrize("ecef, exp_lla",
                        [(np.array([[100], [100], [6356752.31424518]]),
                          np.array([[89.9987], [45], [0]])),
                        (np.array([[-2700628.97971166], [-4292443.61165747], [3855152.80233124]]),
                         np.array([[37.4271], [-122.1764], [16]])),
                        (np.array([[6378137], [0], [0]]),
                         np.array([[0], [0], [0]])),
                        (np.array([[0], [6378137], [0]]),
                         np.array([[0], [90], [0]]))
                        ])
def test_ecef2geodetic(ecef, exp_lla):
    """Test ECEF to LLA (in WGS-84 reference ellipsoid) conversion

    Parameters
    ----------
    ecef : np.ndarray
        ECEF test coordinates, shape: 3x1, units: [m, m, m]
    exp_lla : np.ndarray
        Expected LLA values (obtained using MATLAB's equivalent function)
    """
    lla_orig = ecef_to_geodetic(ecef)
    lla_trans = ecef_to_geodetic(ecef.T)
    ecef_back = geodetic_to_ecef(lla_orig)
    np.testing.assert_array_almost_equal(lla_orig, exp_lla, decimal=2)
    np.testing.assert_array_almost_equal(lla_orig, lla_trans.T)
    np.testing.assert_array_almost_equal(ecef_back, ecef, decimal=0)


def test_local_frame(local_ecef, local_lla):
    """Test equivalent initializations for local frame

    Parameters
    ----------
    local_ecef : np.ndarray
        3x1 shaped ECEF coordinates of NED frame origin
    local_lla : np.ndarray
        3x1 shaped LLA (WGS-84) coordinates of NED frame origin
    """
    local_frame_ecef = LocalCoord.from_ecef(local_ecef)
    local_frame_ecef_t = LocalCoord.from_ecef(local_ecef.T)
    local_frame_lla = LocalCoord.from_ecef(local_lla)
    local_frame_lla_t = LocalCoord.from_ecef(local_lla.T)
    # Checking that the same coordinate conversions are being generated
    np.testing.assert_array_almost_equal(local_frame_ecef.ned_to_ecef_matrix,\
                                        local_frame_ecef_t.ned_to_ecef_matrix)
    np.testing.assert_array_almost_equal(local_frame_lla.ned_to_ecef_matrix, \
                                        local_frame_lla_t.ned_to_ecef_matrix)

@pytest.mark.parametrize("ned, exp_ecef",
                        [(np.array([[100], [0], [0]]),
                          np.array([[-2700596], [-4292392], [3855232]])),
                        (np.array([[0], [100], [0]]),
                         np.array([[-2700544], [-4292496], [3855152]])),
                        (np.array([[0], [0], [100]]),
                         np.array([[-2700586], [-4292376], [3855092]])),
                        (np.array([[0], [0], [0]]),
                         np.array([[-2700628], [-4292443], [3855152]])),
                        (np.array([[100], [100], [100]]),
                         np.array([[-2700469], [-4292378], [3855171]]))
                        ])
def test_ned_conversions(local_frame, ned, exp_ecef):
    """Test NED to ECEF position conversions (and ECEF to NED for same values)

    Parameters
    ----------
    local_frame : gnss_lib_py.utils.coordinates.LocalCoord
        NED local frame of reference initialized for local_lla position
    ned : np.ndarray
        Input NED coordinates
    exp_ecef : np.ndarray
        Expected ECEF values for reference local frame
    """
    ecef = local_frame.ned_to_ecef(ned)
    ecef_t = local_frame.ned_to_ecef(ned.T)
    np.testing.assert_array_almost_equal(ecef, exp_ecef, decimal=0)
    np.testing.assert_array_almost_equal(ecef, ecef_t.T)
    ned_back = local_frame.ecef_to_ned(exp_ecef)
    ned_back_t = local_frame.ecef_to_ned(exp_ecef.T)
    np.testing.assert_array_almost_equal(ned_back, ned, decimal=0)
    np.testing.assert_array_almost_equal(ned_back, ned_back_t.T)


@pytest.mark.parametrize("nedv, exp_ecefv",
                        [(np.array([[100], [0], [0]]),
                          np.array([[32.36], [51.44], [79.41]])),
                        (np.array([[0], [100], [0]]),
                         np.array([[84.64], [-53.25], [0]])),
                        (np.array([[0], [0], [100]]),
                         np.array([[42.28], [67.21], [-60.77]])),
                        (np.array([[0], [0], [0]]),
                         np.array([[0], [0], [0]])),
                        (np.array([[100], [100], [100]]),
                         np.array([[159.29], [65.40], [18.63]]))
                        ])
def test_ned_vector_conversions(local_frame, nedv, exp_ecefv):
    """Test NED to ECEF conversions for vectors (and ECEF to NED for same values)

    Parameters
    ----------
    local_frame : gnss_lib_py.utils.coordinates.LocalCoord
        NED local frame of reference initialized for local_lla position
    nedv : np.ndarray
        Input NED vector values
    exp_ecef : np.ndarray
        Expected ECEF vector values for given reference local frame
    """
    ecefv = local_frame.ned_to_ecefv(nedv)
    ecefv_t = local_frame.ned_to_ecefv(nedv.T)
    np.testing.assert_array_almost_equal(ecefv, exp_ecefv, decimal=0)
    np.testing.assert_array_almost_equal(ecefv, ecefv_t.T)
    nedv_back = local_frame.ecef_to_nedv(exp_ecefv)
    nedv_back_t = local_frame.ecef_to_nedv(exp_ecefv.T)
    np.testing.assert_array_almost_equal(nedv_back, nedv, decimal=0)
    np.testing.assert_array_almost_equal(nedv_back, nedv_back_t.T)


def test_geodetic_to_ned(local_frame):
    """Test conversion from NED to geodetic and back for given NED frame

    Parameters
    ----------
    local_frame : gnss_lib_py.utils.coordinates.LocalCoord
        NED frame of reference initialized at local_lla
    """
    lla = np.array([[38], [-122], [0]])
    exp_ned = np.array([[63598.8877483255], [15494.9039975858], [352.834573334931]])
    ned = local_frame.geodetic_to_ned(lla)
    ned_t = local_frame.geodetic_to_ned(lla.T)
    np.testing.assert_array_almost_equal(ned, exp_ned, decimal=2)
    np.testing.assert_array_almost_equal(ned, ned_t.T)
    lla_back = local_frame.ned_to_geodetic(ned)
    lla_back_t = local_frame.ned_to_geodetic(ned.T)
    np.testing.assert_array_almost_equal(lla_back, lla, decimal=0)
    np.testing.assert_array_almost_equal(lla_back, lla_back_t.T)

@pytest.fixture(name="expected_elaz")
def fixture_expected_elaz():
    """Set the expected elevation and azimuth from sample positions.

    Returns
    -------
    expect_elaz : np.ndarray
        Array containing 6 el/az pairs for testing elaz function

    """
    expect_elaz = np.array([[90.0, -90.0, 0.0 ,  0.0 , 0.0, 0.0  ],
                            [0.0 ,  0.0 , 90.0, -90.0, 0.0, 180.0]]).T
    return expect_elaz


@pytest.fixture(name="set_sv_pos")
def fixture_set_sv_pos():
    """Set the sample satellite positions for computing elevation and azimuth.

    Returns
    -------
    sv_pos : np.ndarray
        Array containing 6 satellite x, y, z coordinates

    """
    sv_pos = np.zeros([6, 3])
    sv_pos[0,0] =  consts.A*1.25
    sv_pos[1,0] =  consts.A*0.75
    sv_pos[2,0] =  consts.A
    sv_pos[2,1] =  consts.A
    sv_pos[3,0] =  consts.A
    sv_pos[3,1] = -consts.A
    sv_pos[4,0] =  consts.A
    sv_pos[4,2] =  consts.A
    sv_pos[5,0] =  consts.A
    sv_pos[5,2] = -consts.A
    return sv_pos

@pytest.fixture(name="set_rx_pos")
def fixture_set_rx_pos():
    """Set the sample reciever position for computing elaz.

    Returns
    -------
    rx_pos : np.ndarray
        Array containing 6 satellite x, y, z coordinates

    """
    rx_pos = np.reshape(np.array([consts.A, 0, 0]), [1, 3])
    return rx_pos

def test_ecef_to_el_az(expected_elaz, set_sv_pos, set_rx_pos):
    """Test receiver to satellite azimuth and elevation calculation.

    Parameters
    ----------
    expected_elaz : fixture
        Expected elevation and azimuth angles generated by the given satellite
        and receiver positions.
    set_sv_pos : fixture
        Satellite position setter
    set_rx_pos : fixture
        Receiver position setter

    """
    calc_elaz = ecef_to_el_az(set_rx_pos, set_sv_pos)
    np.testing.assert_array_almost_equal(expected_elaz, calc_elaz)
