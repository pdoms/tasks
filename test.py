import unittest
import numpy as np
from src.math_ops import *
from src.distances import *



class TestTasks(unittest.TestCase):
    def test_deg_to_rad(self):
        """degree to radian"""
        c_1 = degree_to_radian(45)
        r_1 = 0.7853981634
        self.assertAlmostEqual(c_1, r_1, 3)
        c_2 = degree_to_radian(150)
        r_2 = 2.6179938780
        self.assertAlmostEqual(c_2, r_2, 3)
    def test_rad_to_deg(self):
        """radian to degree"""
        c_1 = int(radian_to_degree(0.7853981634))
        r_1 = 45
        self.assertEqual(c_1, r_1)
        c_2 = int(radian_to_degree(2.6179938780))
        r_2 = 150
        self.assertEqual(c_2, r_2)
    def test_area_trap(self):
        """area of trapezoid"""
        d_1 = area_trapezoid(5,6,5)
        self.assertAlmostEqual(d_1, 27.5, 1)




class TestDistances(unittest.TestCase):
    def test_is_np(self):
        """test f is_np function returns np array"""
        l = [1,2,3]
        test_against = np.array([1,2,3])
        non_np = is_np(l)
        self.assertTrue((non_np == test_against).all())
        self.assertTrue(((is_np(test_against) == test_against)).all())
    
    def test_cosine_distance(self):
        '''test if cosine_distance returns the correct cosine distance'''
        test = np.array([3, 2, 0, 5 ])
        test_2 = np.array([1, 0, 0, 0])
        r = cosine_distance(test, test_2)
        self.assertAlmostEqual(r, 0.487, places=2)




if __name__ == '__main__':
    unittest.main()

