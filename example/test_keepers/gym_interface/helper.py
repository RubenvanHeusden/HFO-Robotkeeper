from __future__ import division
import pandas as pd
import numpy as np
import math


def log_results(results, filename):
    headers = ["goals", "stopped_by_defense", 'out_of_time', 'out_of_bounds']
    

def bin_ball_position(ball_pos, x_bounds, y_bounds,x_size=3, y_size=3):
    x_bins = np.array(np.linspace(x_bounds[0], x_bounds[1], num=x_size))
    y_bins = np.array(np.linspace(y_bounds[0], y_bounds[1], num=y_size))
    
    ball_bin = np.digitize(ball_pos[0], x_bins), np.digitize(ball_pos[1], y_bins)
    return min([ball_bin[0],x_size-1])*10+min([ball_bin[1], y_size-1])


def law_of_cosines(a, b, c):
    return math.acos((a**2+b**2-c**2)/(2*a*b))

def euclid_dist(point1, point2):
    return math.sqrt((point2[0]-point1[0])**2+(point2[1]-point1[1])**2)



def angle_feature(features):
    angles = [0, 0]
    upper_goal_post = (0.85, -0.19)
    lower_goal_post = (0.85,  0.19)
    player_width = 0.035
    ball_pos = features[3:5]
    agent_pos = features[0:2]
    agent_top = [agent_pos[0], agent_pos[1]-player_width]
    agent_bottom = [agent_pos[0], agent_pos[1]+player_width]
    
    grad1 = (agent_top[1]-ball_pos[1]) / (agent_top[0]-ball_pos[0])
    end_pos = [upper_goal_post[0], ((upper_goal_post[0] - agent_top[0])*grad1)+agent_top[1]]
    if end_pos[1] > upper_goal_post[1]:
        ball_top_post_dist = euclid_dist(ball_pos, upper_goal_post)
        ball_end_pos_dist = euclid_dist(ball_pos, end_pos)
        top_post_end_pos_dist = euclid_dist(upper_goal_post, end_pos)
        
        angles[0] = law_of_cosines(ball_top_post_dist, ball_end_pos_dist, top_post_end_pos_dist)
    elif end_pos[1] > lower_goal_post[1]:
        angles[0] = law_of_cosines(euclid_dist(ball_pos, upper_goal_post), euclid_dist(ball_pos, lower_goal_post), euclid_dist(upper_goal_post, lower_goal_post))
    else:
        angles[0] = 0

    grad2 = (agent_bottom[1]-ball_pos[1]) / (agent_bottom[0]-ball_pos[0])
    end_pos = [lower_goal_post[0], ((lower_goal_post[0] - agent_bottom[0])*grad1)+agent_bottom[1]]
    if end_pos[1] < lower_goal_post[1]:
        ball_bottom_post_dist = euclid_dist(ball_pos, lower_goal_post)
        ball_end_pos_dist = euclid_dist(ball_pos, end_pos)
        lower_post_end_pos_dist = euclid_dist(lower_goal_post, end_pos)
        
        angles[1] = law_of_cosines(ball_bottom_post_dist, ball_end_pos_dist, lower_post_end_pos_dist)
    elif end_pos[1] < upper_goal_post[1]:
        angles[1] = law_of_cosines(euclid_dist(ball_pos, upper_goal_post), euclid_dist(ball_pos, lower_goal_post), euclid_dist(upper_goal_post, lower_goal_post))
    else:
        angles[1] = 0
        
    return math.degrees(angles[0]), math.degrees(angles[1])





def build_feature_vector(features):
    
    vec = np.zeros((15, 1))

    # agent's x pos
    vec[0] = features[0]
    # agent's y pos
    vec[1] = features[1]
    # global direction of the agent
    vec[2] = features[2]
    # ball x pos
    vec[3] = features[3]
    # ball y pos
    vec[4] = features[4]
    # proximity of agent to goal
    vec[5] = features[6]
    # angle from agent to goal centre
    vec[6] = features[7]
    # proximity of agent to opponent
    vec[7] = features[9]
    # x pos of opponent
    vec[8] = features[10]
    # y pos of opponent
    vec[9] = features[11]
    # 
    vec[10] = 0
    #
    vec[11] = 0
    #
    vec[12] = 0
    #
    vec[13] = 0
    #
    vec[14] = 0
    
    return vec




    
    

