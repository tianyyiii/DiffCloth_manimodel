import trimesh
import gdist
from tqdm import tqdm
import time
import random
import torch
from torch.autograd.functional import jacobian as torch_jacobian
import numpy as np
import common
from pySim.pySim import pySim, pySimF
import diffcloth_py as diffcloth
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import sys
import traceback
import os
# sys.path.insert(0, "/root/autodl-tmp/DiffCloth_manimodel/pylib")
sys.path.insert(0, os.path.abspath("./pylib"))


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


def jacobian_expand(vertices, triangles, points, jacobian):
    # vertices = np.array(mesh.vertices, dtype=np.float64)
    # triangles = np.array(mesh.faces, dtype=np.int32)
    # distances = gdist.local_gdist_matrix(
    #     vertices, triangles, max_distance=10)
    diff = vertices[:, np.newaxis, :] - vertices[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff**2, axis=-1))
    distances[distances == 0] = np.inf
    mask = np.isin(range(distances.shape[1]), points)
    distances[:, ~mask] = np.inf
    nearest_point = np.argmin(distances, axis=1)
    jacobian_full = np.zeros((30, len(vertices)*3))
    for point in range(len(vertices)):
        if point not in points:
            index = [i for i, v in enumerate(
                points) if v == nearest_point[point]]
            if len(index) == 0:
                index = 0
            else:
                index = index[0]
        else:
            index = points.index(point)
        jacobian_full[:, point*3:point*3+3] = jacobian[:, index*3:index*3+3]
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


def calculate_jacobian_part_test1(pysim, x, v, a, keypoints):
    jacobian = torch.zeros((len(keypoints) * 3, 3))
    a0 = a.clone().detach()
    a0.requires_grad = True
    x1, _ = step(x.clone().detach(), v.clone().detach(), a0, pysim)
    for i, keypoint in enumerate(keypoints):
        for axis in range(3):
            loss = x1[keypoint * 3 + axis]
            loss.backward(retain_graph=True)
            jacobian[i * 3 + axis, :] = a0.grad.data
            a0.grad.zero_()
    return jacobian


def calculate_jacobian_part_test2(pysim, x, v, a, keypoints):
    # 确保加速度向量可以求导
    a0 = a.clone().detach().requires_grad_(True)

    # 定义一个辅助函数，该函数接受加速度并返回选中关键点的位置
    def keypoint_positions(a0):
        x1, _ = step(x.clone().detach(), v.clone().detach(), a0, pysim)
        return torch.cat([x1[keypoint * 3:keypoint * 3 + 3] for keypoint in keypoints])

    # 使用 torch.autograd.functional.jacobian 计算雅可比矩阵
    jacobian_matrix = torch_jacobian(keypoint_positions, a0)

    return jacobian_matrix


def calculate_jacobian(x, v, keypoints, config):
    points = random.sample(range(int(len(x)/3)), 1)
    config = config.copy()
    for index, point in enumerate(tqdm(points)):
        config['scene']['customAttachmentVertexIdx'] = [(0., [point])]
        sim, _, _ = set_sim_from_config(config)
        helper = diffcloth.makeOptimizeHelperWithSim("wear_hat", sim)
        pysim = pySim(sim, helper, True)
        a = x.detach().clone()[point * 3: point * 3 + 3]
        # t1 = time.time()
        # jacobian_part = calculate_jacobian_part(pysim, x, v, a, keypoints)
        # t2 = time.time()
        # jacobian_part_test1 = calculate_jacobian_part_test1(pysim, x, v, a, keypoints)
        # t3 = time.time()
        # jacobian_part_test2 = calculate_jacobian_part_test2(
        #     pysim, x, v, a, keypoints)
        # t4 = time.time()
        jacobian_part = calculate_jacobian_part_test1(pysim, x, v, a, keypoints)
        if index == 0:
            jacobian = jacobian_part
        else:
            jacobian = torch.cat((jacobian, jacobian_part), dim=1)
    return jacobian, points


# def calculate_jacobian_part_parallel(x, v, keypoints, config, point):
#     config = config.copy()
#     config['scene']['customAttachmentVertexIdx'] = [(0., [point])]
#     sim, _, _ = set_sim_from_config(config)
#     helper = diffcloth.makeOptimizeHelperWithSim("wear_hat", sim)
#     pysim = pySim(sim, helper, True)
#     a = x.detach().clone()[point * 3 : point * 3 + 3]
#     return calculate_jacobian_part(pysim, x, v, a, keypoints)


# def calculate_jacobian_parallel(x, v, keypoints, config):
#     points = random.sample(range(int(len(x)/3)), 100)
#     config = config.copy()
#     # with multiprocessing.Pool(processes=20) as pool:
#     #     results = pool.starmap(calculate_jacobian_part_parallel, [(x.detach(), v.detach(), keypoints, config, point) for point in points])
#     with ProcessPoolExecutor(max_workers=1) as executor:
#         futures = [executor.submit(calculate_jacobian_part_parallel, x.clone().detach(), v.clone().detach(), keypoints, config, point) for point in points]
#         results = [future.result() for future in futures]

#     jacobian = torch.cat(results, dim=1)
#     return jacobian, points


def full_jacobian(vertices, triangles, x, v, keypoints, config):
    # mesh = trimesh.load(mesh)
    # x_pos = x.view(-1, 3)
    # mesh.vertices = x_pos.detach().numpy()
    # jacobian, points = calculate_jacobian(x, v, keypoints, config)
    vertices = x.detach().numpy().reshape(-1, 3).astype(np.float64)
    jacobian, points = calculate_jacobian(x, v, keypoints, config)
    jacobian_full = jacobian_expand(vertices, triangles, points, jacobian)
    jacobian_full = torch.tensor(jacobian_full)
    jacobian_full = jacobian_full.view(10, 3, -1, 3).permute(0, 2, 1, 3)
    # repeat_index = random.sample(range(jacobian_full.shape[1]), 2048-jacobian_full.shape[1])
    # jacobian_repeat = jacobian_full[:, repeat_index, :, :]
    # jacobian_full = torch.cat((jacobian_full, jacobian_repeat), dim=1)
    return jacobian_full


def jacobian(mesh, x, v, keypoints, config):
    start = time.time()
    mesh = trimesh.load(mesh)
    x_pos = x.view(-1, 3)
    mesh.vertices = x_pos.detach().numpy()
    jacobian, points = calculate_jacobian(x, v, keypoints, config)
    jacobian_full = jacobian_expand(mesh, points, jacobian)
    jacobian_full = torch.tensor(jacobian_full)
    jacobian_full = jacobian_full.view(10, 3, -1, 3).permute(0, 2, 1, 3)
    repeat_index = random.sample(range(jacobian_full.shape[1]), 2048-jacobian_full.shape[1])
    jacobian_repeat = jacobian_full[:, repeat_index, :, :]
    jacobian_full = torch.cat((jacobian_full, jacobian_repeat), dim=1)
    print(time.time()-start, "jacobian time")
    return jacobian_full

    


    
