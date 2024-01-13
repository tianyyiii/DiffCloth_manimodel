import sys
import os
import argparse

#sys.path.insert(0, "/root/autodl-tmp/DiffCloth_manimodel/pylib")
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
import trajectory
import jacobian
import gc
import math

from renderer import WireframeRenderer
import pywavefront




CONFIG = {
    'fabric': {
        "clothDimX": 6,
        "clothDimY": 6,
        "k_stiff_stretching": 1200,
        "k_stiff_bending":  2,
        "gridNumX": 40,
        "gridNumY": 80,
        "density": 0.2,
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


def mnormal(mesh, x_pos):
    shape = trimesh.load(mesh)
    try:
        vertices = shape.vertices
    except:
        shape = mesh.dump()[1]
        vertices = shape.vertices
    shape.vertices = x_pos
    return shape.vertex_normals


def sample_gaussian(variance=0.25):
    mean = [0, 0, 0]  
    covariance = [[variance, 0, 0],  
                [0, variance, 0],
                [0, 0, variance]]
    sample = np.random.multivariate_normal(mean, covariance)
    return sample


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
    
def trial_forward(x, v, config, mesh_path, trial_step=15):
    x = x.clone().detach()
    v = v.clone().detach()
    config = config.copy()
    attached_points = random.sample(range(int(len(x)/3)), 2)
    config['scene']['customAttachmentVertexIdx'] = [
        (0.0, [attached_points[0], attached_points[1]])]
    sim_in, _, _ = set_sim_from_config(config)
    helper_in = diffcloth.makeOptimizeHelperWithSim("wear_hat", sim_in)
    pysim_in = pySim(sim_in, helper_in, True)
    target_pos = np.zeros(6)
    start_pos = np.concatenate([x[3*i : 3*i+3] for i in attached_points])
    target_pos[:3] = start_pos[:3] + sample_gaussian(0.0001 * trial_step * trial_step)
    target_pos[3:6] = start_pos[3:6] + sample_gaussian(0.0001 * trial_step * trial_step)
    a = np.linspace(start_pos, target_pos, trial_step)
    for a_step in a:
        x, v = step(x, v, torch.tensor(a_step), pysim_in)
    attached_point_final = np.zeros((2, 3))
    attached_point_final[0, :] = x[attached_points[0]*3:attached_points[0]*3+3]
    attached_point_final[1, :] = x[attached_points[1]*3:attached_points[1]*3+3]
    x = x.view(-1, 3).detach().numpy()
    target_normal = mnormal(mesh_path, x)
    return attached_points, attached_point_final, x, target_normal


def main(path, category, on_ground=True, render=False, save=False):
    if category == "hat":
        on_ground = False
    kp_path = "../" + path + "/" + "kp" + ".pcd"
    mesh_path = "../" + path + "/" + "mesh" + ".obj"
    kp_idx = get_keypoints(mesh_path, kp_path)
    config = CONFIG.copy()
    config['fabric']['name'] = "../../../" + mesh_path
    config['scene']['customAttachmentVertexIdx'] = [(0.0, [])]
    sim_init, x0, v0 = set_sim_from_config(config)
    helper_init = diffcloth.makeOptimizeHelperWithSim("wear_hat", sim_init)
    pysim_init = pySim(sim_init, helper_init, True)
    
    N = int(len(x0)/3)
    m = 10
    k = 10
    samples, jacobian_step, T = trajectory.trajectory(category, x0.clone().detach(), kp_idx)
    
    init_state = np.zeros((T, N, 3))
    init_state_normal = np.zeros((T, N, 3))
    response_matrix = np.zeros((T, k, N, 3, 3))
    target_state = np.zeros((T, m, k, 3))
    target_state_normal = np.zeros((T, m, k, 3))
    attached_point_array = np.zeros((T, m, 2))
    attached_point_target = np.zeros((T, m, 2, 3))
    keypoints = np.array(kp_idx)
    frictional_coeff = np.array(0.5)
    kp = np.array(config["fabric"]["k_stiff_stretching"])
    kd = np.zeros(config["fabric"]["k_stiff_bending"])
    if on_ground:
        for i in tqdm.tqdm(range(100)):
            a = torch.tensor([])
            x0, v0 = step(x0, v0, a, pysim_init)
        v0 = v0 * 0
    else:
        config['scene']['primitiveConfig'] = 0
    t = 0
    for i, sample in enumerate(samples):
        x, v = x0.clone().detach(), v0.clone().detach()
        attached_points = sample["attached_points"]
        config['scene']['customAttachmentVertexIdx'] = [
            (0.0, [attached_points[0], attached_points[1]])]
        sim_out, _, _ = set_sim_from_config(config)
        helper_out = diffcloth.makeOptimizeHelperWithSim("wear_hat", sim_out)
        pysim_out = pySim(sim_out, helper_out, True)
        
        motion = sample["motion"]
        num_points = motion.shape[0]
        for j in tqdm.tqdm(range(num_points)):
            if jacobian_step[j] == 1:
                v = v * 0
                response_matrix[t, :, :, :, :] = jacobian.jacobian(mesh_path, x.clone().detach(), v.clone().detach(), kp_idx, config.copy()).detach().numpy()
                #response_matrix[t, :, :, :, :] = np.zeros((k, N, 3, 3))
                x_pos = x.view(-1, 3).detach().numpy()
                init_state[t, :, :] = x_pos  
                init_state_normal[t, :, :] = mnormal(mesh_path, x_pos)
                attached_point_array[t, 0, :] = attached_points
                
                for trial in range(m-1):
                    trial_attach, trial_target, trail_x, trial_normal = trial_forward(x.clone().detach(), v.clone().detach(), config.copy(), mesh_path)
                    attached_point_array[t, trial+1, :] = trial_attach
                    attached_point_target[t, trial+1, :, :] = trial_target
                    target_state[t, trial+1, :, :] = trail_x[kp_idx]
                    target_state_normal[t, trial+1, :, :] = trial_normal[kp_idx]
                    
                if j != 0:
                    attached_point_target[t-1, 0, 0, :] = x[attached_points[0]*3:attached_points[0]*3+3].detach().numpy()
                    attached_point_target[t-1, 0, 1, :] = x[attached_points[1]*3:attached_points[1]*3+3].detach().numpy()
                    target_state[t-1, 0, :, :] = x_pos[kp_idx]
                    target_state_normal[t-1, 0, :, :] = mnormal(mesh_path, x_pos)[kp_idx]
                    
                t += 1
                    
                    
            if j==(num_points-1):
                attached_point_target[t-1, 0, 0, :] = x[attached_points[0]*3:attached_points[0]*3+3].detach().numpy()
                attached_point_target[t-1, 0, 1, :] = x[attached_points[1]*3:attached_points[1]*3+3].detach().numpy()
                target_state[t-1, 0, :, :] = x_pos[kp_idx]
                target_state_normal[t-1, 0, :, :] = mnormal(mesh_path, x_pos)[kp_idx]
                    
            a = torch.tensor(motion[j, :])
            x, v = step(x, v, a, pysim_out)
                
        if render:
            render_record(sim_out)
        if save:
            sim_out.exportCurrentSimulation(category+str(i))
    os.makedirs('/home/ubuntu/middle/' + path, exist_ok=True)
    np.savez_compressed('/home/ubuntu/middle/' + path + '/cloth.npz', init_state=init_state, init_state_normal=init_state_normal, response_matrix=response_matrix, 
            target_state=target_state, target_state_normal=target_state_normal, attached_point=attached_point_array, attached_point_target=attached_point_target, keypoints
            =keypoints, frictional_coeff=frictional_coeff, kp=kp, kd=kd)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    parser.add_argument('--category', type=str)
    args = parser.parse_args()
    main(path = args.path, category=args.category)