import sys
sys.path.insert(0, "/root/autodl-tmp/DiffCloth_manimodel/pylib")
import diffcloth_py as diffcloth
from pySim.pySim import pySim
from pySim.functional import SimFunction

import torch
import numpy as np

import random


def step(x, v, a, simModule):
    x1, v1 = simModule(x, v, a)
    return x1, v1
    
def set_sim_from_config(config):
    sim = diffcloth.makeSimFromConfig(config)
    sim.resetSystem()
    stateInfo = sim.getStateInfo()
    x0 = stateInfo.x
    v0 = stateInfo.v
    x0 = torch.tensor(x0, requires_grad=True)
    v0 = torch.tensor(v0, requires_grad=True)
    return sim, x0, v0

def cubic_bezier(p0, p1, p2, p3, t):
    """Calculate a point on a cubic Bezier curve with given control points at parameter t."""
    return (1-t)**3 * p0 + 3 * (1-t)**2 * t * p1 + 3 * (1-t) * t**2 * p2 + t**3 * p3

def create_bent_curve(p0, p3, bend_factor=0.5, num_points=100):
    """Create a curve that bends towards the z-axis and passes through p0 and p3."""
    # Calculate control points for bending
    p1 = p0 + np.array([0, bend_factor, 0])
    p2 = p3 + np.array([0, bend_factor, 0])

    # Generate points along the curve
    t_values = np.linspace(0, 1, num_points)
    curve_points = np.array([cubic_bezier(p0, p1, p2, p3, t)
                            for t in t_values])
    return curve_points

def discretize_curve(curve):
    jacobian_point = []
    curve = np.vstack((curve[0:-1:20], np.array([curve[-1]])))
    for segment in range(5):
        if segment != 4:
            points = np.linspace(curve[segment], curve[segment+1], int(np.linalg.norm(curve[segment+1]-curve[segment], ord=2)/0.01), endpoint=True) 
        else:
            points = np.linspace(curve[segment], curve[segment+1], int(np.linalg.norm(curve[segment+1]-curve[segment], ord=2)/0.01), endpoint=False)  
        parts = points.shape[0]//10
        jacobian_part = [0] * points.shape[0]
        indexes = np.linspace(0, points.shape[0], parts, endpoint=False)
        for index in indexes:
            jacobian_part[int(index)] = 1
        jacobian_point += jacobian_part
        if segment == 0:
            curve_points = points
        else: 
            curve_points = np.vstack((curve_points, points))
    jacobian_point[-1] = 1
    return curve_points, jacobian_point
        
def sample_gaussian(variance=0.25):
    mean = [0, 0, 0]  
    covariance = [[variance, 0, 0],  
                [0, variance, 0],
                [0, 0, variance]]
    sample = np.random.multivariate_normal(mean, covariance)
    return sample

#This should return a dict of attachedpoints and motion of different sequences
def trajectory(category, x, keypoints, trajectory_num=10, sequence_length=250, jacobian_step=25):
    vertex_num = int(len(x)/3)
    if category == "tops":
        pass
    
    if category == "hat":
        trajectory = []
        line_segment_step = 50
        for index in range(trajectory_num):
            example = dict()
            attached_points = random.sample(range(vertex_num), 2)
            attached_points = np.array(attached_points)
            example["attached_points"] = attached_points
            target_motion = np.zeros((sequence_length, 6))
            for stage in range(5):
                for attach_index in [0, 1]:
                    if stage == 0:
                        start_pos = x[attached_points[attach_index]*3:attached_points[attach_index]*3+3].detach().clone().numpy()
                    else:
                        start_pos = target_motion[stage*line_segment_step-1, attach_index*3:attach_index*3+3]
                    end_pos = start_pos + sample_gaussian()
                    target_motion[stage*line_segment_step:stage*line_segment_step+line_segment_step, attach_index*3:attach_index*3+3] = np.linspace(start_pos, end_pos, num=line_segment_step, endpoint=False)
            example["motion"] = target_motion
            trajectory.append(example)  
            T = int(trajectory_num * sequence_length/jacobian_step)
            jacobian = [1 if i % jacobian_step == 0 else 0 for i in range(sequence_length)]
        return trajectory, jacobian, T
        
    
if __name__ == "__main__":
    p0 = np.array([1, 1, 1])
    p1 = np.array([1, 1, -1])
    curve = create_bent_curve(p0, p1)
    discretize_curve(curve)

        
        
        