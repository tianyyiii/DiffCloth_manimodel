import sys
import os

sys.path.insert(0, os.path.abspath("./pylib"))

import contextlib
import io
import json
import time
import random
import open3d as o3d
import trimesh
import torch
import numpy as np
import tqdm
import common
from pySim.functional import SimFunction
from pySim.pySim import pySim, pySimF
import diffcloth_py as diffcloth
import gc
import math

from renderer import WireframeRenderer
import pywavefront
# sys.path.insert(0, "/root/autodl-tmp/DiffCloth_XMake/pylib")



CONFIG = {
    'fabric': {
        "clothDimX": 6,
        "clothDimY": 6,
        "k_stiff_stretching": 550,
        "k_stiff_bending":  0.01,
        "gridNumX": 40,
        "gridNumY": 80,
        "density": 1,
        "keepOriginalScalePoint": False,
        'isModel': True,
        "custominitPos": False,
        "fabricIdx": 2,  # Enum Value
        "color": (0.3, 0.9, 0.3),
        "name":  "remeshed/top.obj",
    },
    'scene': {
        "orientation": 1,  # Enum Value
        "attachmentPoints": 2,  # CUSTOM_ARRAY
        "customAttachmentVertexIdx": [(0., [])],
        "trajectory": 0,  # Enum Value
        "primitiveConfig": 3,  # Enum Value
        'windConfig': 0,  # Enum Value
        'camPos':  (-10.38, 4.243, 12.72),
        "camFocusPos": (0, -4, 0),
        'camFocusPointType': 3,  # Enum Value
        "sceneBbox":  {"min": (-7, -7, -7), "max": (7, 7, 7)},
        "timeStep": 1.0 / 90.0,
        "stepNum": 250,
        "forwardConvergenceThresh": 1e-8,
        'backwardConvergenceThresh': 5e-4,
        'name': "wind_tshirt"
    }
}


def set_sim_from_config(config):
    sim = diffcloth.makeSimFromConfig(config)
    sim.resetSystem()
    stateInfo = sim.getStateInfo()
    x0 = stateInfo.x
    v0 = stateInfo.v
    x0 = torch.tensor(x0, requires_grad=True)
    v0 = torch.tensor(v0, requires_grad=True)
    return sim, x0, v0


def step(x, v, a, simModule):
    x1, v1 = simModule(x, v, a)
    return x1, v1

def stepF(x, v, a, f, simModule):
    x1, v1 = simModule(x, v, a, f)
    return x1, v1

# def get_keypoints(mesh_file, kp_file):
#     mesh = o3d.io.read_triangle_mesh(mesh_file)
#     mesh_vertices = np.asarray(mesh.vertices)

#     pcd = o3d.io.read_point_cloud(kp_file)
#     pcd_points = np.asarray(pcd.points)

#     indices = []
#     for point in pcd_points:
#         distances = np.sqrt(np.sum((mesh_vertices - point)**2, axis=1))
#         nearest_vertex_index = np.argmin(distances)
#         indices.append(nearest_vertex_index)

#     return indices


def read_mesh_ignore_vtvn(mesh_file):
    pos_vec = []  # 存储顶点位置
    tri_vec = []  # 存储三角形面

    with open(mesh_file, "r") as file:
        for line in file:
            tokens = line.split()
            if not tokens or tokens[0].startswith("#"):
                continue

            if tokens[0] == "v":  # 顶点
                x, y, z = map(float, tokens[1:4])
                pos_vec.append((x, y, z))

            elif tokens[0] == "f":  # 面
                # 仅处理每个面的顶点索引，忽略可能的纹理和法线索引
                vertex_indices = [
                    int(face.partition("/")[0]) - 1 for face in tokens[1:4]
                ]
                tri_vec.append(vertex_indices)

    print("load mesh: ", mesh_file, " with ", len(pos_vec),
          " vertices and ", len(tri_vec), " faces")
    return np.array(pos_vec), np.array(tri_vec)


def get_keypoints(mesh_file, kp_file):
    # mesh = o3d.io.read_triangle_mesh(mesh_file)
    # mesh_vertices = np.asarray(mesh.vertices)
    mesh_vertices, _ = read_mesh_ignore_vtvn(mesh_file)

    pcd = o3d.io.read_point_cloud(kp_file)
    pcd_points = np.asarray(pcd.points)

    indices = []
    for point in pcd_points:
        distances = np.sqrt(np.sum((mesh_vertices - point) ** 2, axis=1))
        nearest_vertex_index = np.argmin(distances)
        indices.append(nearest_vertex_index)

    return indices


def get_coord_by_idx(x, idx):
    return x[idx*3:(idx+1)*3]


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


def render_record(sim, kp_idx=None, curves=None):
    renderer = WireframeRenderer(backend="pyglet")

    forwardRecords = sim.forwardRecords

    mesh_vertices = forwardRecords[0].x.reshape(-1, 3)
    mesh_faces = np.array(diffcloth.getSimMesh(sim))
    x_records = [forwardRecords[i].x.reshape(-1, 3)
                 for i in range(len(forwardRecords))]

    renderer.add_mesh(mesh_vertices, mesh_faces, x_records)
    if kp_idx is not None:
        renderer.add_kp(mesh_vertices, kp_idx)

    if curves is not None:
        for c in curves:
            renderer.add_curve(c)

    renderer.show()
    renderer.run()


def dlg_dress():
    config = CONFIG.copy()
    config['fabric']['name'] = "objs/DLG_Dress032_1.obj"
    config['scene']['customAttachmentVertexIdx'] = [(0.0, [])]
    sim, x0, v0 = set_sim_from_config(config)
    kp_idx = get_keypoints("src/assets/meshes/objs/DLG_Dress032_1.obj",
                           "src/assets/meshes/objs/kp_DLG_Dress032_1.pcd")
    helper = diffcloth.makeOptimizeHelperWithSim("wear_hat", sim)
    pysim = pySim(sim, helper, True)

    # select_kp_idx = [4, 7, 1, 8]
    select_kp_idx = [5, 7, 0, 8]

    for i in tqdm.tqdm(range(100)):
        # stateInfo = sim.getStateInfo()
        # a = torch.tensor(a)
        a = torch.tensor([])
        x0, v0 = step(x0, v0, a, pysim)

    v0 = v0 * 0
    # render_record(sim)

    config['scene']['customAttachmentVertexIdx'] = [
        (0.0, [kp_idx[select_kp_idx[0]], kp_idx[select_kp_idx[2]]])]
    sim, _, _ = set_sim_from_config(config)
    helper = diffcloth.makeOptimizeHelperWithSim("wear_hat", sim)
    pysim = pySim(sim, helper, True)

    p0 = get_coord_by_idx(x0, kp_idx[select_kp_idx[0]])
    p1 = get_coord_by_idx(x0, kp_idx[select_kp_idx[1]])
    p2 = get_coord_by_idx(x0, kp_idx[select_kp_idx[2]])
    p3 = get_coord_by_idx(x0, kp_idx[select_kp_idx[3]])
    # p0 = get_coord_by_idx(x0, kp_idx[5])
    # p1 = get_coord_by_idx(x0, kp_idx[7])
    # p2 = get_coord_by_idx(x0, kp_idx[0])
    # p3 = get_coord_by_idx(x0, kp_idx[8])

    num_points = 100

    curve_points1 = create_bent_curve(p0.detach().numpy(
    ), p1.detach().numpy(), bend_factor=1.5, num_points=num_points)
    curve_points2 = create_bent_curve(p2.detach().numpy(
    ), p3.detach().numpy(), bend_factor=1.5, num_points=num_points)

    for i in tqdm.tqdm(range(num_points)):
        # stateInfo = sim.getStateInfo()
        # a = stateInfo.x_fixedpoints
        # a = a + np.array([0, 0.1, 0])
        # a = torch.tensor(a)
        a = torch.tensor(np.concatenate((curve_points1[i], curve_points2[i])))
        x0, v0 = step(x0, v0, a, pysim)
        v0 = v0 * 0

    render_record(sim, [kp_idx[select_kp_idx[0]], kp_idx[select_kp_idx[1]],
                  kp_idx[select_kp_idx[2]], kp_idx[select_kp_idx[3]]], curves=[curve_points1, curve_points2])


def dlg_dress_force():
    config = CONFIG.copy()
    config['fabric']['name'] = "objs/DLG_Dress032_1.obj"
    config['scene']['customAttachmentVertexIdx'] = [(0.0, [])]
    sim, x0, v0 = set_sim_from_config(config)
    kp_idx = get_keypoints("src/assets/meshes/objs/DLG_Dress032_1.obj",
                           "src/assets/meshes/objs/kp_DLG_Dress032_1.pcd")
    helper = diffcloth.makeOptimizeHelperWithSim("wear_hat", sim)
    pysim = pySim(sim, helper, True)

    # select_kp_idx = [4, 7, 1, 8]
    select_kp_idx = [5, 7, 0, 8]

    for i in tqdm.tqdm(range(100)):
        # stateInfo = sim.getStateInfo()
        # a = torch.tensor(a)
        a = torch.tensor([])
        x0, v0 = step(x0, v0, a, pysim)

    v0 = v0 * 0
    # render_record(sim)

    config['scene']['customAttachmentVertexIdx'] = [
        (0.0, [])]
    sim, _, _ = set_sim_from_config(config)
    helper = diffcloth.makeOptimizeHelperWithSim("wear_hat", sim)
    pysim = pySimF(sim, helper, True)

    p0 = get_coord_by_idx(x0, kp_idx[select_kp_idx[0]])
    p1 = get_coord_by_idx(x0, kp_idx[select_kp_idx[1]])
    p2 = get_coord_by_idx(x0, kp_idx[select_kp_idx[2]])
    p3 = get_coord_by_idx(x0, kp_idx[select_kp_idx[3]])
    # p0 = get_coord_by_idx(x0, kp_idx[5])
    # p1 = get_coord_by_idx(x0, kp_idx[7])
    # p2 = get_coord_by_idx(x0, kp_idx[0])
    # p3 = get_coord_by_idx(x0, kp_idx[8])

    num_points = 5

    curve_points1 = create_bent_curve(p0.detach().numpy(
    ), p1.detach().numpy(), bend_factor=1.5, num_points=num_points)
    curve_points2 = create_bent_curve(p2.detach().numpy(
    ), p3.detach().numpy(), bend_factor=1.5, num_points=num_points)
    
    force_coeff = 180
    max_step_per_points = 20
    distance_threshold = 0.1

    for i in tqdm.tqdm(range(num_points)):
        # stateInfo = sim.getStateInfo()
        # a = stateInfo.x_fixedpoints
        # a = a + np.array([0, 0.1, 0])
        # a = torch.tensor(a)
        # a = torch.tensor(np.concatenate((curve_points1[i], curve_points2[i])))
        f = torch.zeros_like(x0)
        p0_idx = kp_idx[select_kp_idx[0]]
        p2_idx = kp_idx[select_kp_idx[2]]
        p0_now = get_coord_by_idx(x0, p0_idx).detach().numpy()
        p2_now = get_coord_by_idx(x0, p2_idx).detach().numpy()
        
        f[p0_idx*3:(p0_idx+1)*3] = torch.tensor(
            curve_points1[i] - p0_now) * force_coeff
        f[p2_idx*3:(p2_idx+1)*3] = torch.tensor(
            curve_points2[i] - p2_now) * force_coeff

        for j in range(max_step_per_points):
            x0, v0 = stepF(x0, v0, torch.tensor([]), f, pysim)
            
            distance = np.linalg.norm(get_coord_by_idx(x0, p0_idx).detach().numpy() - curve_points1[i])
            distance += np.linalg.norm(get_coord_by_idx(x0, p2_idx).detach().numpy() - curve_points2[i])
            if distance < distance_threshold:
                break
            # v0 = v0 * 0
        v0 = v0 * 0

    render_record(sim, [kp_idx[select_kp_idx[0]], kp_idx[select_kp_idx[1]],
                  kp_idx[select_kp_idx[2]], kp_idx[select_kp_idx[3]]], curves=[curve_points1, curve_points2])


def long_dress():
    config = CONFIG.copy()
    config['fabric']['name'] = "objs/DLLS_dress6.obj"
    config['scene']['customAttachmentVertexIdx'] = [(0.0, [])]
    # config['fabric']['density'] = 0.2
    kp_idx = get_keypoints("src/assets/meshes/objs/DLLS_dress6.obj",
                           "src/assets/meshes/objs/kp_DLLS_dress6.pcd")
    sim, x0, v0 = set_sim_from_config(config)
    helper = diffcloth.makeOptimizeHelperWithSim("wear_hat", sim)
    pysim = pySim(sim, helper, True)

    # select_kp_idx = [5, 7, 0, 8]
    select_kp_idx = [4, 7, 1, 8]

    for i in tqdm.tqdm(range(90)):
        # stateInfo = sim.getStateInfo()
        # a = torch.tensor(a)
        a = torch.tensor([])
        x0, v0 = step(x0, v0, a, pysim)

    # diffcloth.render(sim, False, False)
    render_record(sim)

    config['scene']['customAttachmentVertexIdx'] = [
        (0.0, [kp_idx[select_kp_idx[0]], kp_idx[select_kp_idx[2]]])]
    sim, _, v0 = set_sim_from_config(config)
    helper = diffcloth.makeOptimizeHelperWithSim("wear_hat", sim)
    pysim = pySim(sim, helper, True)

    p0 = get_coord_by_idx(x0, kp_idx[select_kp_idx[0]])
    p1 = get_coord_by_idx(x0, kp_idx[select_kp_idx[1]])
    p2 = get_coord_by_idx(x0, kp_idx[select_kp_idx[2]])
    p3 = get_coord_by_idx(x0, kp_idx[select_kp_idx[3]])
    # p0 = get_coord_by_idx(x0, kp_idx[5])
    # p1 = get_coord_by_idx(x0, kp_idx[7])
    # p2 = get_coord_by_idx(x0, kp_idx[0])
    # p3 = get_coord_by_idx(x0, kp_idx[8])

    num_points = 400

    curve_points1 = create_bent_curve(
        p0.detach().numpy(), p1.detach().numpy(), bend_factor=3, num_points=num_points)
    curve_points2 = create_bent_curve(
        p2.detach().numpy(), p3.detach().numpy(), bend_factor=3, num_points=num_points)

    for i in tqdm.tqdm(range(num_points)):
        # stateInfo = sim.getStateInfo()
        # a = stateInfo.x_fixedpoints
        # a = a + np.array([0, 0.1, 0])
        # a = torch.tensor(a)
        a = torch.tensor(np.concatenate((curve_points1[i], curve_points2[i])))
        x0, v0 = step(x0, v0, a, pysim)

    render_record(sim, [kp_idx[select_kp_idx[0]], kp_idx[select_kp_idx[1]],
                  kp_idx[select_kp_idx[2]], kp_idx[select_kp_idx[3]]], curves=[curve_points1, curve_points2])


if __name__ == '__main__':
    dlg_dress_force()
    # dlg_dress()
    # long_dress()
