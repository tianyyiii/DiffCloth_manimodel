import sys
sys.path.insert(0, "/root/autodl-tmp/DiffCloth_XMake/pylib")

import diffcloth_py as diffcloth
from pySim.pySim import pySim
from pySim.functional import SimFunction
import common

import numpy as np
import torch
import trimesh
import open3d as o3d
import random
import time
import json
import os 
import io
import contextlib

CONFIG = {
    'fabric': {
        "clothDimX" : 6,
        "clothDimY" : 6,
        "k_stiff_stretching" : 10000, 
        "k_stiff_bending" :  0.1,
        "gridNumX" : 40,
        "gridNumY" : 80,
        "density" : 0.3,
        "keepOriginalScalePoint" : False,
        'isModel' : True,
        "custominitPos" : False,
        "fabricIdx" : 1,  # Enum Value
        "color" : (0.9, 0.9, 0.9),
        "name" :  "remeshed/top.obj",
    },
    'scene' : {
        "orientation" : 0, # Enum Value
        "attachmentPoints" : 2, # CUSTOM_ARRAY
        "customAttachmentVertexIdx": [(0., [])], 
        "trajectory" : 0, # Enum Value
        "primitiveConfig" : 3, # Enum Value
        'windConfig' : 0, # Enum Value
        'camPos' :  (-10.38, 4.243, 12.72),
        "camFocusPos" : (0, 0, 0),
        'camFocusPointType' : 3, # Enum Value
        "sceneBbox" :  {"min": (-7, -7, -7), "max": (7, 7, 7)},
        "timeStep" : 1.0 / 90.0,
        "stepNum" : 250,
        "forwardConvergenceThresh" : 1e-8,
        'backwardConvergenceThresh' : 5e-4,
        'name' : "wind_tshirt"
    }
}


def step(x, v, a, simModule):
    x1, v1 = simModule(x, v, a)
    return x1, v1
    
def set_sim(example):
    sim = diffcloth.makeSim(example)
    sim.resetSystem()
    stateInfo = sim.getStateInfo()
    x0 = stateInfo.x
    v0 = stateInfo.v
    x0 = torch.tensor(x0, requires_grad=True)
    v0 = torch.tensor(v0, requires_grad=True)
    return sim, x0, v0

def set_sim_from_config(config):
    sim = diffcloth.makeSimFromConfig(config)
    sim.resetSystem()
    stateInfo = sim.getStateInfo()
    x0 = stateInfo.x
    v0 = stateInfo.v
    x0 = torch.tensor(x0, requires_grad=True)
    v0 = torch.tensor(v0, requires_grad=True)
    return sim, x0, v0
    


def calculate_jacobian_part(pysim, x0, v0, a0, keypoints):
    total_forward_time = 0
    forward_iteration = 0
    total_backward_time = 0
    backward_iteration = 0
    jacobian = torch.zeros((len(keypoints) * 3, 3))
    for i, keypoint in enumerate(keypoints):
        for axis in range(3):
            a00 = a0.clone().detach()
            a00.requires_grad = True
            time_start = time.time()
            x1, v1 = step(x0.clone().detach(), v0.clone().detach(), a00, pysim)
            time_end = time.time()
            total_forward_time += (time_end - time_start)
            forward_iteration += 1
            loss = x1[keypoint * 3 + axis]
            time_start = time.time()
            loss.backward()
            time_end = time.time()
            total_backward_time += (time_end - time_start)
            backward_iteration += 1
            jacobian[i * 3 + axis, :] = a00.grad
    print("forward_time", total_forward_time/forward_iteration)
    print("backward_time", total_backward_time/backward_iteration)
    return jacobian

def calculate_jacobian(x0, v0, keypoints):
    points = random.sample(range(int(len(x0)/3)), 100)
    print(points)
    for index, point in enumerate(points):
        config = CONFIG
        config['scene']['customAttachmentVertexIdx'] = [(0., [point])]
        sim, x0, v0 = set_sim_from_config(config)
        helper = diffcloth.makeOptimizeHelper(example)
        pysim = pySim(sim, helper, True)
        a = sim.getStateInfo().x_fixedpoints
        a = torch.tensor(a)
        jacobian_part = calculate_jacobian_part(pysim, x0, v0, a, keypoints)
        if index == 0:
            jacobian = jacobian_part
        else:
            jacobian = torch.cat((jacobian, jacobian_part), dim=1)
    return jacobian, points
            
        
    
def init_cloth(sim, x0, v0, show_detail=False):
    x, v = x0, v0
    z_mean = []
    for i in range(400):
        a = sim.getStateInfo().x_fixedpoints
        a = torch.tensor(a)
        x, v = step(x, v, a, pysim)
        z_mean.append(sum(x[1:-1:3])/len(x[1:-1:3]))
    if show_detail:
        sim.exportCurrentSimulation("manimodel")
        print("z_mean_trajectory", z_mean)
    return x, v

def one_step_sample_trajectory(x, v, trajectory, show_detail=False):
    if trajectory == "class_agnostic_one_point":
        data = []
        axis_1 = x[0:-1:3]
        axis_2 = x[1:-1:3]
        axis_3 = x[2:-1:3]
        sigma_1 = (max(axis_1)-min(axis_1))/2
        sigma_2 = (max(axis_2)-min(axis_2))/2
        sigma_3 = (max(axis_3)-min(axis_3))/2
        for index in range(1000):
            attached_point = random.sample(range(int(len(x)/3)), 1)[0]
            config = CONFIG
            config['scene']['customAttachmentVertexIdx'] = [(0., [attached_point])]
            sim, x0, v0 = set_sim_from_config(config)
            pysim = pySim(sim, helper, True)
            a = x[attached_point*3:attached_point*3+3]
            attached_points_0 = a.tolist()
            attached_points_0 = [attached_points_0[i:i + 3] for i in range(0, len(attached_points_0), 3)]
            print("----------------------------")
            print(a, "a_before")
            a[0] += random.gauss(0, sigma_1**(1/2))
            a[1] += random.gauss(0, sigma_2**(1/2))
            a[2] += random.gauss(0, sigma_3**(1/2))
            print(a, "a_after")
            x_1, v_1 = step(x.clone().detach(), v.clone().detach(), a, pysim)
            
            sequence = dict()
            x_start = x.detach().numpy().tolist()
            points_start = [x_start[i:i + 3] for i in range(0, len(x_start), 3)]
            x_target = x_1.detach().numpy().tolist()
            points_target = [x_target[i:i + 3] for i in range(0, len(x_target), 3)]
            sequence["init_state"] = points_start
            sequence["target_state"] = points_target
            attached_points = x[attached_point*3:attached_point*3+3].tolist()
            attached_points = [attached_points[i:i + 3] for i in range(0, len(attached_points), 3)]
            sequence["attached_point"] = attached_points_0
            sequence["attached_point_target"] = attached_points
            sequence["response_matrix"] = np.zeros((len(x_start), 3 * 10)).tolist()
            for i in sequence:
                print(np.array(sequence[i]).shape)
            data.append(sequence)
            if show_detail:
                os.makedirs(f"/root/autodl-tmp/DiffCloth_XMake/output/sample/{str(index)}", exist_ok=True)
                sim.exportCurrentSimulation("sample" + "/" + str(index))
            
        with open('data_demo.json', 'w') as json_file:
            json.dump(data, json_file)
            
            
def get_keypoints(mesh_file, kp_file):
    mesh = o3d.io.read_triangle_mesh(mesh_file)
    mesh_vertices = np.asarray(mesh.vertices)

    pcd = o3d.io.read_point_cloud(kp_file)
    pcd_points = np.asarray(pcd.points)

    indices = []
    for point in pcd_points:
        distances = np.sqrt(np.sum((mesh_vertices - point)**2, axis=1))
        nearest_vertex_index = np.argmin(distances)
        indices.append(nearest_vertex_index)
        
    return indices
     
        
        

if __name__ == "__main__":
    torch.set_printoptions(precision=8)
    common.setRandomSeed(1349)
    example = "wear_hat"
    sim, x0, v0 = set_sim_from_config(CONFIG)
    helper = diffcloth.makeOptimizeHelper(example)
    pysim = pySim(sim, helper, True)
    
    keypoints = get_keypoints("TNLC_shirt2.obj", "kp_TNLC_shirt2.pcd")
    keypoint_pos = np.zeros((10, 3))
    x_temp = x0.detach().numpy()
    for i in range(10):
        keypoint_pos[i, :] = x_temp[keypoints[i] * 3 : keypoints[i] * 3 + 3]
    keypoint_pos = keypoint_pos.tolist()
    jacobian, points = calculate_jacobian(x0, v0, keypoints)
    point_pos = np.zeros((100, 3))
    for i in range(10):
        point_pos[i, :] = x_temp[points[i] * 3 : points[i] * 3 + 3]
    point_pos = point_pos.tolist()
    
    jaco = dict()
    jaco["jacobian"] = jacobian.detach().numpy().tolist()
    jaco["keypoints"] = keypoint_pos
    jaco["points"] = point_pos
    with open('jacobian.json', 'w') as json_file:
        json.dump(jaco, json_file)
    
    
    #x, v = init_cloth(sim, x0, v0, show_detail=False)
    x, v = x0, v0
    
    one_step_sample_trajectory(x.clone().detach(), v.clone().detach(), "class_agnostic_one_point", False)
    
    # a = x[392*3: 392*3+3]
    # for i in range(5):
    #     x, v = step(x, v, a, pysim)
    # sim.exportCurrentSimulation("fold")
    

    
    
    
    