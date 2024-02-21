import unittest
# from unittest import mock
from unittest.mock import patch
from .IRAF_Reduction import BJD_TDB


class TestBJDTDB(unittest.TestCase):
    def test_BJD_TDB1(self):
        bjd = BJD_TDB("2458403.58763", "lapalma", "00:28:27.97", "78:57:42.66")
        self.assertEqual(bjd.value, 2458403.588447444)
        
    def test_BJD_TDB2(self):
        bjd = BJD_TDB("2458403.58763", "bsuo", "00:28:27.97", "78:57:42.66")
        self.assertEqual(bjd.value, 2458403.588447444)
        
    def test_BJD_TDB3(self):
        bjd = BJD_TDB("2457143.76136", "kpno", "13:27:50.47", "75:55:16.60")
        self.assertEqual(bjd.value, 2457143.762132256)
        
    def test_BJD_TDB4(self):
        bjd = BJD_TDB("2457143.76136", "bsuo", "13:27:50.47", "75:55:16.60")
        self.assertEqual(bjd.value, 2457143.762132256)


unittest.main()
