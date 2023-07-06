import pandas as pd
import tempfile
import unittest
from unittest.mock import patch
from .apass import *
from .vseq_updated import conversion, decimal_limit


class TestGetUserInputs(unittest.TestCase):
    @patch('builtins.input', side_effect=['10:00:00.0000', '10:00:00.0000'])
    def test_get_user_inputs_1(self):
        ra, dec = get_user_inputs()
        self.assertEqual(ra, '10:00:00.0000')
        self.assertEqual(dec, '10:00:00.0000')

    @patch('builtins.input', side_effect=['20:00:00.0000', '20:00:00.0000'])
    def test_get_user_inputs_2(self):
        ra, dec = get_user_inputs()
        self.assertEqual(ra, '20:00:00.0000')
        self.assertEqual(dec, '20:00:00.0000')


class TestQueryVizier(unittest.TestCase):
    @patch('apass.query_vizier', autospec=True)
    def test_query_vizier_1(self, mock_vizier):
        mock_vizier_instance = mock_vizier.return_value
        mock_vizier_instance.query_region.return_value = {"II/336/apass9": "fake_table"}

        ra_input = '10:00:00.0000'
        dec_input = '10:00:00.0000'
        result = query_vizier(ra_input, dec_input)
        self.assertEqual(result, "fake_table")

    @patch('apass.query_vizier', autospec=True)
    def test_query_vizier_2(self, mock_vizier):
        mock_vizier_instance = mock_vizier.return_value
        mock_vizier_instance.query_region.return_value = {"II/336/apass9": "another_fake_table"}

        ra_input = '20:00:00.0000'
        dec_input = '20:00:00.0000'
        result = query_vizier(ra_input, dec_input)
        self.assertEqual(result, "another_fake_table")


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
        vizier_result = [
            [30, 40, 1, 0.1, 2, 0.2, 3, 0.3, 4, 0.4],
            [60, 80, 2, 0.2, 3, 0.3, 4, 0.4, 5, 0.5]
        ]  # replace this with the actual structure of the Vizier data you receive

        expected_df = pd.DataFrame({
            "RA": conversion([30 / 15, 60 / 15]),
            "Dec": conversion([40, 80]),
            "Bmag": decimal_limit([1, 2]),
            "e_Bmag": decimal_limit([0.1, 0.2]),
            "Vmag": decimal_limit([2, 3]),
            "e_Vmag": decimal_limit([0.2, 0.3]),
            "g'mag": decimal_limit([3, 4]),
            "e_g'mag": decimal_limit([0.3, 0.4]),
            "r'mag": decimal_limit([4, 5]),
            "e_r'mag": decimal_limit([0.4, 0.5])
        })

        actual_df = process_data(vizier_result)
        pd.testing.assert_frame_equal(actual_df, expected_df)

    def test_process_data_2(self):
        vizier_result = [
            [90, 100, 3, 0.3, 4, 0.4, 5, 0.5, 6, 0.6],
            [120, 140, 4, 0.4, 5, 0.5, 6, 0.6, 7, 0.7]
        ]  # replace this with the actual structure of the Vizier data you receive

        expected_df = pd.DataFrame({
            "RA": conversion([90 / 15, 120 / 15]),
            "Dec": conversion([100, 140]),
            "Bmag": decimal_limit([3, 4]),
            "e_Bmag": decimal_limit([0.3, 0.4]),
            "Vmag": decimal_limit([4, 5]),
            "e_Vmag": decimal_limit([0.4, 0.5]),
            "g'mag": decimal_limit([5, 6]),
            "e_g'mag": decimal_limit([0.5, 0.6]),
            "r'mag": decimal_limit([6, 7]),
            "e_r'mag": decimal_limit([0.6, 0.7])
        })

        actual_df = process_data(vizier_result)
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
        mock_splitter.return_value = [1.39319167]  # Expected decimal equivalent for "1:23:45.67"
        mock_query.return_value = "Vizier result"
        mock_process.return_value = pd.DataFrame({"data": [1, 2, 3]})
        mock_input.return_value = "test_file.txt"
        expected = ("test_file.txt", 1.39319167, "-12:34:56.7")

        actual = catalog_finder()
        self.assertEqual(actual, expected)
        mock_save.assert_called_once_with(mock_process.return_value, mock_input.return_value)

    @patch('apass.get_user_inputs')
    @patch('vseq_updated.splitter')
    @patch('apass.query_vizier')
    @patch('apass.process_data')
    @patch('builtins.input')
    @patch('apass.save_to_file')
    def test_catalog_finder_2(self, mock_save, mock_process, mock_query, mock_splitter, mock_get_inputs):
        ra = "1:23:45.67"
        dec = "-12:34:56.7"
        pipeline = True
        folder_path = "folder"
        obj_name = "obj_name"
        mock_splitter.return_value = [1.39319167, -12.58241667]  # Expected decimal equivalents for ra and dec
        mock_query.return_value = "Vizier result"
        mock_process.return_value = pd.DataFrame({"data": [1, 2, 3]})
        expected = (folder_path + "\\APASS_" + obj_name + ".txt", 1.39319167, -12.58241667)

        actual = catalog_finder(ra, dec, pipeline, folder_path, obj_name)
        self.assertEqual(actual, expected)
        mock_save.assert_called_once_with(mock_process.return_value, expected[0])


class TestCreateLines(unittest.TestCase):
    @patch('apass.angle_dist')
    @patch('vseq_updated.splitter')
    def test_create_lines_no_angle_dist(self, mock_splitter, mock_angle_dist):
        mock_splitter.return_value = ['10', '20']
        mock_angle_dist.return_value = False
        ra_list = ['10', '20']
        dec_list = ['30', '40']
        mag_list = ['50', '60']
        ra = '10'
        dec = '30'
        filt = 'V'

        result = create_lines(ra_list, dec_list, mag_list, ra, dec, filt)
        expected = '10, 30, 1, 1, 50\n20, 40, 1, 1, 60\n'

        self.assertEqual(result, expected)

    @patch('apass.angle_dist')
    @patch('vseq_updated.splitter')
    def test_create_lines_with_angle_dist(self, mock_splitter, mock_angle_dist):
        mock_splitter.side_effect = [['10', '20'], ['30', '40']]
        mock_angle_dist.return_value = True
        ra_list = ['10', '20']
        dec_list = ['30', '40']
        mag_list = ['50', '60']
        ra = '10'
        dec = '30'
        filt = 'V'

        result = create_lines(ra_list, dec_list, mag_list, ra, dec, filt)
        expected = '20, 40, 1, 1, 60\n'

        self.assertEqual(result, expected)


class TestCreateRADEC(unittest.TestCase):
    @patch('apass.create_header')
    @patch('apass.create_lines')
    @patch('builtins.input')
    @patch('builtins.open')
    def test_create_radec_no_pipeline(self, mock_open, mock_input, mock_create_lines, mock_create_header):
        df = pd.DataFrame({"data": [1, 2, 3]})
        ra = "1:23:45.67"
        dec = "-12:34:56.7"
        T_list = [4, 5, 6]
        pipeline = False
        folder_path = ""
        obj_name = ""

        mock_create_header.return_value = "header"
        mock_create_lines.side_effect = ["lines1", "lines2", "lines3", "lines4"]
        mock_input.side_effect = ["output1", "output2", "output3", "output4"]
        mock_file = MagicMock()
        mock_open.side_effect = [mock_file, mock_file, mock_file, mock_file]

        create_radec(df, ra, dec, T_list, pipeline, folder_path, obj_name)

        mock_create_header.assert_called_once_with(ra, dec)
        mock_create_lines.assert_has_calls([
            call(df[1], df[2], df[3], ra, dec, "B"),
            call(df[1], df[2], df[5], ra, dec, "V"),
            call(df[1], df[2], df[7], ra, dec, "R"),
            call(df[1], df[2], T_list, ra, dec, "T")
        ])
        mock_input.assert_has_calls([
            call("Please enter an output file pathway \033[1m\033[93mWITHOUT\033[00m the extension but with the file name for the B filter RADEC file, for AIJ (i.e. C:\\folder1\\folder2\[filename]): "),
            call("Please enter an output file pathway \033[1m\033[93mWITHOUT\033[00m the extension but with the file name for the V filter RADEC file, for AIJ (i.e. C:\\folder1\\folder2\[filename]): "),
            call("Please enter an output file pathway \033[1m\033[93mWITHOUT\033[00m the extension but with the file name for the R filter RADEC file, for AIJ (i.e. C:\\folder1\\folder2\[filename]): "),
            call("Please enter an output file pathway \033[1m\033[93mWITHOUT\033[00m the extension but with the file name for the T filter RADEC file, for AIJ (i.e. C:\\folder1\\folder2\[filename]): ")
        ])
        mock_open.assert_has_calls([
            call("output1.radec", "w"),
            call("output2.radec", "w"),
            call("output3.radec", "w"),
            call("output4.radec", "w")
        ])
        mock_file.write.assert_has_calls([
            call("headerlines1"),
            call("headerlines2"),
            call("headerlines3"),
            call("headerlines4")
        ])

    @patch('apass.create_header')
    @patch('apass.create_lines')
    @patch('builtins.open', new_callable=mock_open)
    @patch('builtins.print')
    def test_create_radec_no_pipeline(self, mock_print, mock_open, mock_create_lines, mock_create_header):
        df = pd.DataFrame({"data": [1, 2, 3]})
        ra = "1:23:45.67"
        dec = "-12:34:56.7"
        T_list = [4, 5, 6]
        pipeline = False
        folder_path = "folder"
        obj_name = "obj_name"

        mock_create_header.return_value = "header"
        mock_create_lines.side_effect = ["lines1", "lines2", "lines3", "lines4"]

        with patch('builtins.input', side_effect=["file1", "file2", "file3", "file4"]):
            create_radec(df, ra, dec, T_list, pipeline, folder_path, obj_name)

        mock_create_header.assert_called_once_with(ra, dec)
        mock_create_lines.assert_has_calls([
            call(df[1], df[2], df[3], ra, dec, "B"),
            call(df[1], df[2], df[5], ra, dec, "V"),
            call(df[1], df[2], df[7], ra, dec, "R"),
            call(df[1], df[2], T_list, ra, dec, "T")
        ])
        mock_print.assert_called_with("\nFinished writing RADEC files for Johnson B, Johnson V, Cousins R, and T.\n")

        # Assert the contents of the written files
        mock_open.assert_has_calls([
            call("file1.radec", "w"),
            call("file2.radec", "w"),
            call("file3.radec", "w"),
            call("file4.radec", "w")
        ])

        file_handles = [handle[0] for handle in mock_open.mock_calls]
        file_contents = [handle.write.call_args[0][0] for handle in file_handles]

        expected_contents = [
            "headerlines1",
            "headerlines2",
            "headerlines3",
            "headerlines4"
        ]

        for content, expected in zip(file_contents, expected_contents):
            assert content == expected

    @patch('apass.create_header')
    @patch('apass.create_lines')
    @patch('builtins.open', new_callable=mock_open)
    @patch('builtins.print')
    def test_create_radec_large_df_no_pipeline(self, mock_print, mock_open, mock_create_lines, mock_create_header):
        df = pd.DataFrame({
            "RA": [1.234, 2.345, 3.456, 4.567, 5.678],
            "Dec": [-12.345, -23.456, -34.567, -45.678, -56.789],
            "Bmag": [10.0, 11.0, 12.0, 13.0, 14.0],
            "Vmag": [9.0, 10.0, 11.0, 12.0, 13.0],
            "Rmag": [8.0, 9.0, 10.0, 11.0, 12.0],
        })
        ra = "1:23:45.67"
        dec = "-12:34:56.7"
        T_list = [15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0]
        pipeline = False
        folder_path = "folder"
        obj_name = "obj_name"

        mock_create_header.return_value = "header"
        mock_create_lines.side_effect = ["linesB", "linesV", "linesR", "linesT"]

        create_radec(df, ra, dec, T_list, pipeline, folder_path, obj_name)

        mock_create_header.assert_called_once_with(ra, dec)
        mock_create_lines.assert_has_calls([
            call(df["RA"], df["Dec"], df["Bmag"], ra, dec, "B"),
            call(df["RA"], df["Dec"], df["Vmag"], ra, dec, "V"),
            call(df["RA"], df["Dec"], df["Rmag"], ra, dec, "R"),
            call(df["RA"], df["Dec"], T_list, ra, dec, "T")
        ])
        mock_print.assert_called_with("\nFinished writing RADEC files for Johnson B, Johnson V, Cousins R, and T.\n")

        # Assert the contents of the written files
        mock_open.assert_has_calls([
            call(folder_path + "\\" + obj_name + "_B.radec", "w"),
            call(folder_path + "\\" + obj_name + "_V.radec", "w"),
            call(folder_path + "\\" + obj_name + "_R.radec", "w"),
            call(folder_path + "\\" + obj_name + "_T.radec", "w")
        ])

        file_handles = [handle[0] for handle in mock_open.mock_calls]
        file_contents = [handle.write.call_args[0][0] for handle in file_handles]

        expected_contents = [
            "headerlinesB",
            "headerlinesV",
            "headerlinesR",
            "headerlinesT"
        ]

        for content, expected in zip(file_contents, expected_contents):
            assert content == expected

    @patch('apass.create_header')
    @patch('apass.create_lines')
    @patch('builtins.open', new_callable=mock_open)
    @patch('builtins.print')
    def test_create_radec_large_df_pipeline(self, mock_print, mock_open, mock_create_lines, mock_create_header):
        df = pd.DataFrame({
            "RA": [1.234, 2.345, 3.456, 4.567, 5.678],
            "Dec": [-12.345, -23.456, -34.567, -45.678, -56.789],
            "Bmag": [10.0, 11.0, 12.0, 13.0, 14.0],
            "Vmag": [9.0, 10.0, 11.0, 12.0, 13.0],
            "Rmag": [8.0, 9.0, 10.0, 11.0, 12.0],
        })
        ra = "1:23:45.67"
        dec = "-12:34:56.7"
        T_list = [15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0]
        pipeline = True
        folder_path = "folder"
        obj_name = "obj_name"

        mock_create_header.return_value = "header"
        mock_create_lines.side_effect = ["linesB", "linesV", "linesR", "linesT"]

        create_radec(df, ra, dec, T_list, pipeline, folder_path, obj_name)

        mock_create_header.assert_called_once_with(ra, dec)
        mock_create_lines.assert_has_calls([
            call(df["RA"], df["Dec"], df["Bmag"], ra, dec, "B"),
            call(df["RA"], df["Dec"], df["Vmag"], ra, dec, "V"),
            call(df["RA"], df["Dec"], df["Rmag"], ra, dec, "R"),
            call(df["RA"], df["Dec"], T_list, ra, dec, "T")
        ])
        mock_print.assert_called_with("\nFinished writing RADEC files for Johnson B, Johnson V, Cousins R, and T.\n")

        # Assert the contents of the written files
        mock_open.assert_has_calls([
            call(folder_path + "\\" + obj_name + "_B.radec", "w"),
            call(folder_path + "\\" + obj_name + "_V.radec", "w"),
            call(folder_path + "\\" + obj_name + "_R.radec", "w"),
            call(folder_path + "\\" + obj_name + "_T.radec", "w")
        ])

        file_handles = [handle[0] for handle in mock_open.mock_calls]
        file_contents = [handle.write.call_args[0][0] for handle in file_handles]

        expected_contents = [
            "headerlinesB",
            "headerlinesV",
            "headerlinesR",
            "headerlinesT"
        ]

        for content, expected in zip(file_contents, expected_contents):
            assert content == expected


class TestCalculations(unittest.TestCase):
    def test_calculations(self):
        i = "15.0"
        V = ["13.0", "14.0", "15.0", "16.0"]
        g = ["14.0", "15.0", "16.0", "17.0"]
        r = ["12.0", "13.0", "14.0", "15.0"]
        gamma = 1.0
        beta = 0.5
        e_beta = 0.1
        alpha = 2.0
        e_alpha = 0.2
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
        gamma = 1.0
        beta = 0.0  # Simulate zero division
        e_beta = 0.1
        alpha = 2.0
        e_alpha = 0.2
        e_B = ["0.1", "0.2", "0.3", "0.4"]
        e_V = ["0.2", "0.3", "0.4", "0.5"]
        e_g = ["0.3", "0.4", "0.5", "0.6"]
        e_r = ["0.4", "0.5", "0.6", "0.7"]
        count = 2

        with self.assertRaises(ZeroDivisionError):
            calculations(i, V, g, r, gamma, beta, e_beta, alpha, e_alpha, e_B, e_V, e_g, e_r, count)

    def test_calculations_with_negative_root(self):
        i = "15.0"
        V = ["13.0", "14.0", "15.0", "16.0"]
        g = ["14.0", "15.0", "16.0", "17.0"]
        r = ["12.0", "13.0", "14.0", "15.0"]
        gamma = 1.0
        beta = -0.5  # Simulate negative root
        e_beta = 0.1
        alpha = 2.0
        e_alpha = 0.2
        e_B = ["0.1", "0.2", "0.3", "0.4"]
        e_V = ["0.2", "0.3", "0.4", "0.5"]
        e_g = ["0.3", "0.4", "0.5", "0.6"]
        e_r = ["0.4", "0.5", "0.6", "0.7"]
        count = 2

        result = calculations(i, V, g, r, gamma, beta, e_beta, alpha, e_alpha, e_B, e_V, e_g, e_r, count)
        expected_root = -1.0  # Negative root
        expected_val = float(V[count]) + (alpha * (float(i) - float(V[count])) - gamma - float(g[count]) + float(r[count])) / beta

        self.assertEqual(result[0], expected_root)
        self.assertEqual(result[1], expected_val)


class TestAngleDist(unittest.TestCase):
    def test_angle_dist_same_position(self):
        x1 = 10.0
        y1 = 20.0
        x2 = 10.0
        y2 = 20.0

        result = angle_dist(x1, y1, x2, y2)
        expected = True

        self.assertEqual(result, expected)

    def test_angle_dist_different_position(self):
        x1 = 10.0
        y1 = 20.0
        x2 = 30.0
        y2 = 40.0

        result = angle_dist(x1, y1, x2, y2)
        expected = False

        self.assertEqual(result, expected)

    def test_angle_dist_close_position(self):
        x1 = 10.0
        y1 = 20.0
        x2 = 10.001
        y2 = 20.001

        result = angle_dist(x1, y1, x2, y2)
        expected = True

        self.assertEqual(result, expected)


class TestAngleDist(unittest.TestCase):
    def test_angle_dist_same_position(self):
        x1 = 10.0
        y1 = 20.0
        x2 = 10.0
        y2 = 20.0

        result = angle_dist(x1, y1, x2, y2)
        expected = True

        self.assertEqual(result, expected)

    def test_angle_dist_different_position(self):
        x1 = 10.0
        y1 = 20.0
        x2 = 30.0
        y2 = 40.0

        result = angle_dist(x1, y1, x2, y2)
        expected = False

        self.assertEqual(result, expected)

    def test_angle_dist_close_position(self):
        x1 = 10.0
        y1 = 20.0
        x2 = 10.001
        y2 = 20.001

        result = angle_dist(x1, y1, x2, y2)
        expected = True

        self.assertEqual(result, expected)


if __name__ == '__main__':
    unittest.main()
