from __future__ import division
import pandas as pd
import numpy as np
import math


def log_results(results, filename):
    headers = ["goals", "stopped_by_defense", 'out_of_time', 'out_of_bounds']
    

def bin_ball_position(ball_pos, x_bounds, y_bounds):
    x_bins = np.array(np.linspace(x_bounds[0], x_bounds[1], num=10))
    y_bins = np.array(np.linspace(y_bounds[0], y_bounds[1], num=10))
    
    ball_bin = np.digitize(ball_pos[0], x_bins), np.digitize(ball_pos[1], y_bins)
    return min([ball_bin[0],9])*10+min([ball_bin[1], 9])


def law_of_cosines(a, b, c):
    return math.acos2((a**2+b**2-c**2)/2*a*b)
    
