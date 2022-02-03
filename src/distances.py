import numpy as np


def is_np(x):
    """makes sure input array(s) are numpy arrays"""
    if isinstance(x, list):
        return np.array(x)
    else: 
        return x

def cosine_distance(x,y):
    dot = np.dot(x, y)
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)
    return dot / (norm_x*norm_y)
    







