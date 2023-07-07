"""
Test functions for apass.py

Author: Kyle Koeller
Last Edited By: Kyle Koeller

Created: 07/07/2021
Last Updated: 07/07/2021
"""

import tempfile
import unittest
# from unittest import mock
from unittest.mock import patch
from apass import *
import numpy as np
import pandas as pd
from astropy.table import Table
# import numpy.testing as np_testing


class TestGetUserInputs(unittest.TestCase):
    @patch('builtins.input', side_effect=['10:00:00.0000', '10:00:00.0000'])
    def test_get_user_inputs_1(self, mock_inputs):
        ra, dec = get_user_inputs()
        self.assertEqual(ra, '10:00:00.0000')
        self.assertEqual(dec, '10:00:00.0000')

    @patch('builtins.input', side_effect=['20:00:00.0000', '20:00:00.0000'])
    def test_get_user_inputs_2(self, mock_inputs):
        ra, dec = get_user_inputs()
        self.assertEqual(ra, '20:00:00.0000')
        self.assertEqual(dec, '20:00:00.0000')


class TestSaveToFile(unittest.TestCase):
    @patch('builtins.print')
    def test_save_to_file_1(self, mock_print):
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        save_to_file(df, temp_file.name)
        mock_print.assert_called_with("\nCompleted save.\n")

        # Read the file back and compare to the original DataFrame
        df_saved = pd.read_csv(temp_file.name)
        pd.testing.assert_frame_equal(df, df_saved)

    @patch('builtins.print')
    def test_save_to_file_2(self, mock_print):
        df = pd.DataFrame({"C": [7, 8, 9], "D": [10, 11, 12]})
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        save_to_file(df, temp_file.name)
        mock_print.assert_called_with("\nCompleted save.\n")

        # Read the file back and compare to the original DataFrame
        df_saved = pd.read_csv(temp_file.name)
        pd.testing.assert_frame_equal(df, df_saved)


class TestProcessData(unittest.TestCase):
    def test_process_data_1(self):
        vizier_result = [[149.80650600, 9.77727500, 11.978, 0.059, 12.795, 0.074, 12.297, 0.051, 11.745, 0.025],
                         [149.92585800, 9.85430400, 12.97, 0.057, 13.620, 0.064, 13.216, 0.044, 12.786, 0.026],
                         [150.05965100, 9.93060200, 12.093, 0.068, 12.774, 0.075, 12.353, 0.063, 11.901, 0.018],
                         [149.97517100, 9.97032400, 12.635, 0.053, 13.478, 0.070, 12.979, 0.035, 12.386, 0.028],
                         [149.88790700, 9.95414100, 11.22, 0.063, 11.853, 0.068, 11.457, 0.038, 11.052, 0.036],
                         [149.85290900, 10.04760400, 11.412, 0.060, 12.578, 0.060, 11.941, 0.046, 11.026, 0.025],
                         [149.79449900, 10.07322900, 10.718, 0.040, 11.922, 0.071, 11.246, 0.036, 10.366, 0.019],
                         [149.86394500, 10.09965200, 11.512, 0.051, 12.646, 0.072, 12.018, 0.049, 11.143, 0.016],
                         [149.77909600, 10.13327500, 12.141, 0.037, 12.842, 0.077, 12.400, 0.035, 11.975, 0.022],
                         [150.21555200, 9.93702300, 11.674, 0.060, 12.274, 0.082, 11.879, 0.040, 11.507, 0.014],
                         [150.11300300, 9.93905100, 13.03, 0.056, 13.900, 0.074, 13.396, 0.043, 12.737, 0.032],
                         [150.05995800, 10.02434500, 10.889, 0.064, 11.635, 0.075, 11.196, 0.050, 10.685, 0.046],
                         [150.18622200, 10.22239600, 12.757, 0.057, 13.495, 0.077, 13.045, 0.041, 12.550, 0.026],
                         [149.86439000, 10.14400900, 12.13, 0.047, 13.095, 0.065, 12.543, 0.039, 11.830, 0.022],
                         [149.86666600, 10.16969700, 11.651, 0.046, 12.281, 0.068, 11.875, 0.033, 11.482, 0.025]]

        columns = list(map(list, zip(*vizier_result)))

        expected = Table(columns, names=(
            '_RAJ2000', '_DEJ2000', 'Vmag', 'e_Vmag', 'Bmag', 'e_Bmag', "g_mag", "e_g_mag", "r_mag", "e_r_mag"))

        expected_df = pd.DataFrame({
            "RA": [
                '9:59:13.561', '9:59:42.206', '10:0:14.316', '9:59:54.041', '9:59:33.098',
                '9:59:24.698', '9:59:10.680', '9:59:27.347', '9:59:6.983', '10:0:51.732',
                '10:0:27.121', '10:0:14.390', '10:0:44.693', '9:59:27.454'
            ],
            "Dec": [
                '9:46:38.190', '9:51:15.494', '9:55:50.167', '9:58:13.166', '9:57:14.908',
                '10:2:51.374', '10:4:23.624', '10:5:58.747', '10:7:59.790', '9:56:13.283',
                '9:56:20.584', '10:1:27.642', '10:13:20.626', '10:8:38.432'
            ],
            "Vmag": [
                11.98, 12.97, 12.09, 12.63, 11.22, 11.41, 10.72, 11.51, 12.14, 11.67,
                13.03, 10.89, 12.76, 12.13
            ],
            "e_Vmag": [
                0.06, 0.06, 0.07, 0.05, 0.06, 0.06, 0.04, 0.05, 0.04, 0.06, 0.06, 0.06, 0.06, 0.05
            ],
            "Bmag": [
                12.79, 13.62, 12.77, 13.48, 11.85, 12.58, 11.92, 12.65, 12.84, 12.27,
                13.90, 11.63, 13.49, 13.10
            ],
            "e_Bmag": [
                0.07, 0.06, 0.07, 0.07, 0.07, 0.06, 0.07, 0.07, 0.08, 0.08, 0.07, 0.07, 0.08, 0.07
            ],
            "g'mag": [
                12.30, 13.22, 12.35, 12.98, 11.46, 11.94, 11.25, 12.02, 12.40, 11.88,
                13.40, 11.20, 13.04, 12.54
            ],
            "e_g'mag": [
                0.05, 0.04, 0.06, 0.04, 0.04, 0.05, 0.04, 0.05, 0.04, 0.04, 0.04, 0.05, 0.04, 0.04
            ],
            "r'mag": [
                11.74, 12.79, 11.90, 12.39, 11.05, 11.03, 10.37, 11.14, 11.97, 11.51,
                12.74, 10.69, 12.55, 11.83
            ],
            "e_r'mag": [
                0.03, 0.03, 0.02, 0.03, 0.04, 0.03, 0.02, 0.02, 0.02, 0.01, 0.03, 0.05, 0.03, 0.02
            ]
        })

        actual_df = process_data(expected)
        actual_df["Vmag"] = actual_df["Vmag"].astype(float)
        actual_df["e_Vmag"] = actual_df["e_Vmag"].astype(float)
        actual_df["Bmag"] = actual_df["Bmag"].astype(float)
        actual_df["e_Bmag"] = actual_df["e_Bmag"].astype(float)
        actual_df["g'mag"] = actual_df["g'mag"].astype(float)
        actual_df["e_g'mag"] = actual_df["e_g'mag"].astype(float)
        actual_df["r'mag"] = actual_df["r'mag"].astype(float)
        actual_df["e_r'mag"] = actual_df["e_r'mag"].astype(float)

        pd.testing.assert_frame_equal(actual_df, expected_df)

    def test_process_data_2(self):
        vizier_result = [
            [227.670746, 14.941402, 11.97, 0.02, 12.86, 0.04, 12.40, 0.04, 11.68, 0.03],
            [227.751509, 15.035514, 11.33, 0.03, 11.98, 0.03, 11.61, 0.05, 11.16, 0.03],
            [227.80151, 15.028085, 12.88, 0.03, 13.47, 0.02, 13.15, 0.03, 12.74, 0.03],
            [227.825468, 15.080871, 12.72, 0.02, 13.29, 0.05, 12.96, 0.05, 12.59, 0.04],
            [227.452095, 15.085105, 10.15, 0.03, 10.76, 0.03, 10.54, 0.18, 10.07, 0.08],
            [227.705792, 15.101218, 13.31, 0.03, 13.94, 0.05, 13.57, 0.04, 13.14, 0.05],
            [227.398479, 15.110404, 10.92, 0.03, 11.44, 0.04, 11.12, 0.04, 10.78, 0.03],
            [227.348021, 15.275232, 12.14, 0.01, 12.68, 0.04, 12.34, 0.04, 12.00, 0.02],
            [227.392271, 15.31242, 12.14, 0.02, 12.70, 0.04, 12.35, 0.04, 11.99, 0.02],
            [227.806202, 15.372171, 12.05, 0.02, 12.60, 0.04, 12.28, 0.05, 11.93, 0.05],
            [227.709975, 15.375034, 10.88, 0.02, 11.83, 0.03, 11.31, 0.04, 10.61, 0.06]
        ]

        columns = list(map(list, zip(*vizier_result)))

        expected = Table(columns, names=(
            '_RAJ2000', '_DEJ2000', 'Vmag', 'e_Vmag', 'Bmag', 'e_Bmag', "g_mag", "e_g_mag", "r_mag", "e_r_mag"))

        expected_df = pd.DataFrame({
            "RA": ['15:10:40.979', '15:11:0.362', '15:11:12.362', '15:11:18.112', '15:9:48.503', '15:10:49.390',
                   '15:9:35.635', '15:9:23.525', '15:9:34.145', '15:11:13.488']
            ,
            "Dec": ['14:56:29.047', '15:2:7.850', '15:1:41.106', '15:4:51.136', '15:5:6.378', '15:6:4.385',
                    '15:6:37.454', '15:16:30.835', '15:18:44.712', '15:22:19.816']
            ,
            "Vmag": [11.97, 11.33, 12.88, 12.72, 10.15, 13.31, 10.92, 12.14, 12.14, 12.05]
            ,
            "e_Vmag": [0.02, 0.03, 0.03, 0.02, 0.03, 0.03, 0.03, 0.01, 0.02, 0.02]
            ,
            "Bmag": [12.86, 11.98, 13.47, 13.29, 10.76, 13.94, 11.44, 12.68, 12.70, 12.60]
            ,
            "e_Bmag": [0.04, 0.03, 0.02, 0.05, 0.03, 0.05, 0.04, 0.04, 0.04, 0.04]
            ,
            "g'mag": [12.40, 11.61, 13.15, 12.96, 10.54, 13.57, 11.12, 12.34, 12.35, 12.28]
            ,
            "e_g'mag": [0.04, 0.05, 0.03, 0.05, 0.18, 0.04, 0.04, 0.04, 0.04, 0.05]
            ,
            "r'mag": [11.68, 11.16, 12.74, 12.59, 10.07, 13.14, 10.78, 12.00, 11.99, 11.93]
            ,
            "e_r'mag": [0.03, 0.03, 0.03, 0.04, 0.08, 0.05, 0.03, 0.02, 0.02, 0.05]

        })

        actual_df = process_data(expected)
        actual_df["Vmag"] = actual_df["Vmag"].astype(float)
        actual_df["e_Vmag"] = actual_df["e_Vmag"].astype(float)
        actual_df["Bmag"] = actual_df["Bmag"].astype(float)
        actual_df["e_Bmag"] = actual_df["e_Bmag"].astype(float)
        actual_df["g'mag"] = actual_df["g'mag"].astype(float)
        actual_df["e_g'mag"] = actual_df["e_g'mag"].astype(float)
        actual_df["r'mag"] = actual_df["r'mag"].astype(float)
        actual_df["e_r'mag"] = actual_df["e_r'mag"].astype(float)

        pd.testing.assert_frame_equal(actual_df, expected_df)


class TestCatalogFinder(unittest.TestCase):
    @patch('apass.get_user_inputs')
    @patch('vseq_updated.splitter')
    @patch('apass.query_vizier')
    @patch('apass.process_data')
    @patch('builtins.input')
    @patch('apass.save_to_file')
    def test_catalog_finder_1(self, mock_save, mock_input, mock_process, mock_query, mock_splitter, mock_get_inputs):
        mock_get_inputs.return_value = ("1:23:45.67", "-12:34:56.7")
        mock_splitter.return_value = [1.3958333]  # Expected decimal equivalent for "1:23:45.67"
        mock_query.return_value = "Vizier result"
        mock_process.return_value = pd.DataFrame({"data": [1, 2, 3]})
        mock_input.return_value = "test_file.txt"
        expected = ("test_file.txt", 1.3958333, -12.5822222)

        actual = catalog_finder()
        self.assertEqual(actual, expected)
        mock_save.assert_called_once_with(mock_process.return_value, mock_input.return_value)

    @patch('apass.get_user_inputs')
    @patch('vseq_updated.splitter')
    @patch('apass.query_vizier')
    @patch('apass.process_data')
    @patch('builtins.input')
    @patch('apass.save_to_file')
    def test_catalog_finder_2(self, mock_save, mock_input, mock_process, mock_query, mock_splitter, mock_get_inputs):
        ra = "1:23:45.67"
        dec = "-12:34:56.7"
        pipeline = True
        folder_path = "folder"
        obj_name = "obj_name"
        mock_splitter.return_value = [1.3958333, -12.5822222]  # Expected decimal equivalents for ra and dec
        mock_query.return_value = "Vizier result"
        mock_process.return_value = pd.DataFrame({"data": [1, 2, 3]})
        expected = (folder_path + "\\APASS_" + obj_name + ".txt", 1.3958333, -12.5822222)

        actual = catalog_finder(ra, dec, pipeline, folder_path, obj_name)
        self.assertEqual(actual, expected)
        mock_save.assert_called_once_with(mock_process.return_value, expected[0])


class TestAngleDist(unittest.TestCase):
    def test_angle_dist_same_position(self):
        # Test when the coordinates are the same position
        x1, y1 = 10.0, 20.0
        x2, y2 = 10.0, 20.0

        comp = angle_dist(x1, y1, x2, y2)

        self.assertEqual(comp, True)

    def test_angle_dist_different_positions(self):
        # Test when the coordinates are different positions
        x1, y1 = 10.0, 20.0
        x2, y2 = 10.0, 30.0

        comp = angle_dist(x1, y1, x2, y2)

        self.assertEqual(comp, False)


class TestCreateLines(unittest.TestCase):
    def test_create_lines_no_angle_dist(self):
        ra_list = ['20:00:00', '50:00:00']
        dec_list = ['40:00:00', '50:00:00']
        mag_list = [12, 10]
        ra = '10'
        dec = '30'
        filt = 'V'

        result = create_lines(ra_list, dec_list, mag_list, ra, dec, filt)
        expected = '20:00:00, 40:00:00, 1, 1, 12\n50:00:00, 50:00:00, 1, 1, 10\n'

        self.assertEqual(expected, result)

    def test_create_lines_with_angle_dist(self):
        ra_list = ['10:00:00', '20:00:00', '30:00:00']
        dec_list = ['30:00:00', '40:00:00', '50:00:00']
        mag_list = [12, 10, 11]
        ra = '10'
        dec = '30'
        filt = 'V'

        result = create_lines(ra_list, dec_list, mag_list, ra, dec, filt)
        expected = '20:00:00, 40:00:00, 1, 1, 10\n30:00:00, 50:00:00, 1, 1, 11\n'
        self.assertEqual(expected, result)


class TestCalculations(unittest.TestCase):
    def test_calculations(self):
        i = "15.0"
        V = ["13.0", "14.0", "15.0", "16.0"]
        g = ["14.0", "15.0", "16.0", "17.0"]
        r = ["12.0", "13.0", "14.0", "15.0"]
        alpha = 0.278
        e_alpha = 0.016
        beta = 1.321
        e_beta = 0.03
        gamma = 0.219
        e_B = ["0.1", "0.2", "0.3", "0.4"]
        e_V = ["0.2", "0.3", "0.4", "0.5"]
        e_g = ["0.3", "0.4", "0.5", "0.6"]
        e_r = ["0.4", "0.5", "0.6", "0.7"]
        count = 2

        result = calculations(i, V, g, r, gamma, beta, e_beta, alpha, e_alpha, e_B, e_V, e_g, e_r, count)
        expected_root = np.sqrt(
            (np.abs((alpha * (float(i) - float(V[count]))) / beta) ** 2 + float(e_V[count]) ** 2 +
             (np.abs((alpha * (float(i) - float(V[count]))) / beta) * np.sqrt(
                 (e_alpha / alpha) ** 2 + ((np.sqrt(float(e_B[count]) ** 2 + float(e_V[count]) ** 2)) /
                                           (float(i) - float(V[count]))) ** 2) / beta) ** 2))
        expected_val = float(V[count]) + (
                alpha * (float(i) - float(V[count])) - gamma - float(g[count]) + float(r[count])) / beta

        self.assertEqual(result[0], expected_root)
        self.assertEqual(result[1], expected_val)

    def test_calculations_with_zero_division(self):
        i = "15.0"
        V = ["13.0", "14.0", "15.0", "16.0"]
        g = ["14.0", "15.0", "16.0", "17.0"]
        r = ["12.0", "13.0", "14.0", "15.0"]
        alpha = 0.278
        e_alpha = 0.016
        beta = 0
        e_beta = 0.03
        gamma = 0.219
        e_B = ["0.1", "0.2", "0.3", "0.4"]
        e_V = ["0.2", "0.3", "0.4", "0.5"]
        e_g = ["0.3", "0.4", "0.5", "0.6"]
        e_r = ["0.4", "0.5", "0.6", "0.7"]
        count = 2

        with self.assertRaises(ZeroDivisionError):
            calculations(i, V, g, r, gamma, beta, e_beta, alpha, e_alpha, e_B, e_V, e_g, e_r, count)
