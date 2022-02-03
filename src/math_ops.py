#exercises taken from https://www.w3resource.com/python-exercises/math/

import math 

def degree_to_radian(degrees):
    '''function to convert degree to radian'''
    return degrees * (math.pi/180)

def radian_to_degree(radian):
    '''function to convert degree to radian'''
    return radian * 180/math.pi

def area_trapezoid(a, b, h):
    '''function to calculate the area of a trapezoid'''
    return ((a+b) / 2) * h