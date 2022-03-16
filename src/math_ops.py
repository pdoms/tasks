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



def vec_sub(v, w):
    return [i - j for i,j in zip(v,w)]

def magnitude(v):
    return math.sqrt(sum(i**2 for i in v))

def dot(v,w):
    return sum(i*j for i,j in zip(v,w))

def distance(v, w):
    return magnitude(vec_sub(v,w))

def cross(v,w):
    return [
        v[1]*w[2] - v[2]*w[1],
        v[2]*w[0] - v[0]*w[2],
        v[0]*w[1] - v[1]*w[0]
    ]

