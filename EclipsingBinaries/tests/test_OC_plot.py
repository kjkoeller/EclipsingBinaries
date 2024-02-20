import unittest
# from unittest import mock
from unittest.mock import patch
from .OC_plot import calculate_oc, arguments


class TestCalculateOC(unittest.TestCase):
    def test_Calculate_OC1(self):
        e, OC, OC_err, _, _ = calculate_oc(10, 0.1, 0, 0.1, 1)
        self.assertEqual(e, 0.5)
        self.assertEqual(OC, '-0.50000')
        self.assertEqual(OC_err, "0.14142")
    
    def test_Calcualte_OC2(self):
        e, OC, OC_err, _, _ = calculate_oc(15.0, 0.1, 5.0, 0.1, 2.0)
        self.assertEqual(e, 5.0)
        self.assertEqual(OC, '0.00000')
        self.assertEqual(OC_err, "0.14142")
    
    def test_Calcualte_OC3(self):
        # line 13 of the test_minimums.txt in the examples folder
        e, OC, OC_err, _, _ = calculate_oc(2458843.932122, 0.00058, 2457143.761819, 0.00014, 0.31297)
        self.assertEqual(e, 5432.5)
        self.assertEqual(OC, '-0.03922')
        self.assertEqual(OC_err, "0.00060")

class TestArguments(unittest.TestCase):
    @patch('builtins.input', side_effect=["10.0", "0.02", "0.312"])
    def test_arguments1(self, mock_inputs):
        result = arguments()
        self.assertEqual(result[0], 10)
        self.assertEqual(result[1], 0.02)
        self.assertEqual(result[2], 0.312)
    
    @patch('builtins.input', side_effect=["100", "0.02", "-0.312"])
    def test_arguments2(self, mock_inputs):
        result = arguments()
        self.assertEqual(result[0], 100.0)
        self.assertEqual(result[1], 0.02)
        self.assertEqual(result[2], -0.312)


unittest.main()
