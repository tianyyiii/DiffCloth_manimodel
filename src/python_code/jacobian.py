import sys
sys.path.insert(0, "/root/autodl-tmp/DiffCloth_manimodel/pylib")

import diffcloth_py as diffcloth
from pySim.pySim import pySim, pySimF
import common

import numpy as np
import torch
import random
import time

from tqdm import tqdm
import gdist
import trimesh

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


def jacobian_expand(mesh, points, jacobian):
    vertices = np.array(mesh.vertices, dtype=np.float64)
    triangles = np.array(mesh.faces, dtype=np.int32)
    distances = gdist.local_gdist_matrix(vertices, triangles, max_distance=10.0)
    distances[distances==0] = np.inf
    mask = np.isin(range(distances.shape[1]), points)
    distances[:, ~mask] = np.inf
    nearest_point = np.argmin(distances, axis=1)
    jacobian_full = np.zeros((30, len(mesh.vertices)*3))
    for point in range(len(mesh.vertices)):
        if point not in points:
            index = [i for i, v in enumerate(points) if v == nearest_point[point, 0]][0]
        else:
            index = points.index(point)
        jacobian_full[:, point*3:point*3+3] = jacobian[:, index*3:index*3+3]
            
    temp = nearest_point[3, 0]
    return jacobian_full


def calculate_jacobian_part(pysim, x, v, a, keypoints):
    jacobian = torch.zeros((len(keypoints) * 3, 3))
    for i, keypoint in enumerate(keypoints):
        for axis in range(3):
            a0 = a.clone().detach()
            a0.requires_grad = True
            x1, _ = step(x.clone().detach(), v.clone().detach(), a0, pysim)
            loss = x1[keypoint * 3 + axis]
            loss.backward()
            jacobian[i * 3 + axis, :] = a0.grad
    return jacobian


def calculate_jacobian(x, v, keypoints, config):
    points = random.sample(range(int(len(x)/3)), 10)
    for index, point in tqdm(enumerate(points)):
        config['scene']['customAttachmentVertexIdx'] = [(0., [point])]
        sim, _, _ = set_sim_from_config(config)
        helper = diffcloth.makeOptimizeHelperWithSim("wear_hat", sim)
        pysim = pySim(sim, helper, True)
        a = x.detach().clone()[point * 3 : point * 3 + 3]
        jacobian_part = calculate_jacobian_part(pysim, x, v, a, keypoints)
        if index == 0:
            jacobian = jacobian_part
        else:
            jacobian = torch.cat((jacobian, jacobian_part), dim=1)
    return jacobian, points

def jacobian(mesh, x, v, keypoints, config):
    start = time.time()
    mesh = trimesh.load(mesh)
    x_pos = x.view(-1, 3)
    mesh.vertices = x_pos.detach().numpy()
    jacobian, points = calculate_jacobian(x, v, keypoints, config)
    jacobian_full = jacobian_expand(mesh, points, jacobian)
    jacobian_full = torch.tensor(jacobian_full)
    jacobian_full = jacobian_full.view(10, 3, -1, 3).permute(0, 2, 1, 3)
    # repeat_index = random.sample(range(jacobian_full.shape[1]), 2048-jacobian_full.shape[1])
    # jacobian_repeat = jacobian_full[:, repeat_index, :, :]
    # jacobian_full = torch.cat((jacobian_full, jacobian_repeat), dim=1)
    print(time.time()-start, "jacobian time")
    return jacobian_full

    


    
