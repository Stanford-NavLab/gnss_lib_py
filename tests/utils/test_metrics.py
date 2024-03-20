"""Test functions for implemented accuracy and quality metrics
"""

__authors__ = "Ashwin Kanhere"
__date__ = "30 January, 2024"


import numpy as np
import pytest

from gnss_lib_py.navdata.navdata import NavData
from gnss_lib_py.utils.metrics import accuracy_statistics


@pytest.fixture(name="rng")
def fixture_rng():
    seed = 0
    rng = np.random.default_rng(seed)
    return rng


@pytest.fixture(name="num_samples")
def fixture_num_samples():
    num_samples = 100
    return num_samples


@pytest.fixture(name="ground_truth")
def fixture_ground_truth(rng, num_samples):
    ground_truth = rng.random((3, num_samples))
    return ground_truth

@pytest.fixture(name="offset")
def fixture_offset():
    offset = [10, 20, 30]
    return offset


@pytest.fixture(name="constant_offset_estimate")
def fixture_constant_offset_estimate(ground_truth, offset):
    constant_offset_estimate = ground_truth.copy()
    for row_idx in range(ground_truth.shape[0]):
        constant_offset_estimate[row_idx] = ground_truth[row_idx] + offset[row_idx]
    return constant_offset_estimate


def array_to_navdata(array, row_names):
    assert len(row_names) == array.shape[0], \
                        "The number of rows in names and array must match"
    data = NavData()
    data['gps_millis'] = np.arange(0, array.shape[1])
    for row_idx, row_name in enumerate(row_names):
        data[row_name] = array[row_idx, :]
    return data

@pytest.mark.parametrize("est_type, se_row_names, gt_row_names",
                         [
                            ("pos", ["x_rx_m", "y_rx_m", "z_rx_m"],
                             ["x_rx_gt_m", "y_rx_gt_m", "z_rx_gt_m"]),
                            ("vel", ["vx_rx_mps", "vy_rx_mps", "vz_rx_mps"],
                             ["vx_rx_gt_mps", "vy_rx_gt_mps", "vz_rx_gt_mps"]),
                            ("acc", ["ax_rx_mps2", "ay_rx_mps2", "az_rx_mps2"],
                             ["ax_rx_gt_mps2", "ay_rx_gt_mps2", "az_rx_gt_mps2"])
                         ])
@pytest.mark.parametrize("statistic",
                         [
                            "mean",
                            "median",
                            "max_min",
                            "percentile",
                            "quantiles",
                            "mean_absolute",
                            "max_absolute",
                          ])
@pytest.mark.parametrize("direction",
                         [
                            None,
                            "ned",
                            "enu",
                            "3d_norm",
                            "horizontal",
                         ])
def test_metrics_same_estimate(ground_truth, est_type, se_row_names,
                               gt_row_names, statistic, direction):
    state_estimate = array_to_navdata(ground_truth, se_row_names)
    ground_truth = array_to_navdata(ground_truth, gt_row_names)
    statistics = accuracy_statistics(state_estimate, ground_truth,
                                     est_type=est_type,
                                     statistic=statistic,
                                     direction=direction,
                                     ecef_origin=np.array([[-2700628], [-4292443], [3855152]]))
    for calc_stat in statistics.values():
        assert calc_stat == 0, \
            "The error should be zero when the estimate and truth are the same"


@pytest.mark.parametrize("est_type, se_row_names, gt_row_names",
                         [
                            ("pos", ["x_rx_m", "y_rx_m", "z_rx_m"],
                             ["x_rx_gt_m", "y_rx_gt_m", "z_rx_gt_m"]),
                            # ("vel", ["vx_rx_mps", "vy_rx_mps", "vz_rx_mps"],
                            #  ["vx_rx_gt_mps", "vy_rx_gt_mps", "vz_rx_gt_mps"]),
                            # ("acc", ["ax_rx_mps2", "ay_rx_mps2", "az_rx_mps2"],
                            #  ["ax_rx_gt_mps2", "ay_rx_gt_mps2", "az_rx_gt_mps2"])
                         ])
@pytest.mark.parametrize("statistic",
                         [
                            "mean",
                            # "median",
                            # "max_min",
                            # "percentile",
                            # "quantiles",
                            # "mean_absolute",
                            # "max_absolute",
                          ])
@pytest.mark.parametrize("direction",
                         [
                            None,
                            # "ned",
                            # "enu",
                            # "3d_norm",
                            # "horizontal",
                         ])
def test_metrics_offset_estimate(ground_truth, constant_offset_estimate,
                                 num_samples, offset, est_type,
                                 se_row_names, gt_row_names,
                                 statistic, direction):
    state_estimate = array_to_navdata(constant_offset_estimate, se_row_names)
    ground_truth = array_to_navdata(ground_truth, gt_row_names)
    statistics = accuracy_statistics(state_estimate, ground_truth,
                                     est_type=est_type,
                                     statistic=statistic,
                                     direction=direction,
                                     ecef_origin=np.array([[-2700628], [-4292443], [3855152]]))
    print('statistics', statistics)
    print('offset', offset)
    for stat_idx, calc_stat in enumerate(statistics.values()):
        if direction == "3d_norm":
            assert calc_stat == np.sqrt(np.sum(np.square(offset)))
        elif direction == "horizontal":
            assert calc_stat == np.sqrt(np.sum(np.square(offset[0:2])))
        else:
            if statistic == "mean":
                if stat_idx%2==0:
                    offset_idx = stat_idx//2
                    assert calc_stat == offset[offset_idx]
                else:
                    assert calc_stat == 0, \
                        "Covariance must be zero for constant offset"
            else:
                assert calc_stat == offset[stat_idx]


def test_ned_metrics_no_origin(ground_truth):
    with pytest.raises(Exception):
        ground_truth_nav = array_to_navdata(ground_truth, ["vx_rx_mps", "vy_rx_mps", "vz_rx_mps"])
        accuracy_statistics(ground_truth_nav, ground_truth_nav,
                            est_type="vel",
                            direction="ned")
    ecef_origin = np.array([[-2700628], [-4292443], [3855152]])
    pos_ground_truth = ground_truth.copy()
    pos_ground_truth[0, :] += ecef_origin[0, 0]
    pos_ground_truth[1, :] += ecef_origin[1, 0]
    pos_ground_truth[2, :] += ecef_origin[2, 0]
    pos_ground_truth = array_to_navdata(pos_ground_truth, ["x_rx_m", "y_rx_m", "z_rx_m"])
    _ = accuracy_statistics(pos_ground_truth, pos_ground_truth,
                            est_type="pos",
                            statistic="mean",
                            direction="ned")


def test_metrics_along_cross_track(ground_truth):
    with pytest.raises(NotImplementedError):
        ground_truth_nav = array_to_navdata(ground_truth, ["x_rx_m", "y_rx_m", "z_rx_m"])
        accuracy_statistics(ground_truth_nav, ground_truth_nav, est_type="pos",
                            direction="along_cross_track")
    with pytest.raises(Exception):
        ground_truth_nav = array_to_navdata(ground_truth, ["x_rx_m", "y_rx_m", "z_rx_m"])
        accuracy_statistics(ground_truth_nav, ground_truth_nav, est_type="pos",
                            direction="along_cross_track")



def test_metrics_unimplemented(ground_truth):
    with pytest.raises(ValueError):
        ground_truth_nav = array_to_navdata(ground_truth, ["x_rx_m", "y_rx_m", "z_rx_m"])
        accuracy_statistics(ground_truth_nav, ground_truth_nav, est_type="pos",
                            statistic="some_extra_statistic")
    with pytest.raises(ValueError):
        ground_truth_nav = array_to_navdata(ground_truth, ["x_rx_m", "y_rx_m", "z_rx_m"])
        accuracy_statistics(ground_truth_nav, ground_truth_nav, est_type="pos",
                            statistic="mean",
                            direction="some_extra_direction")
