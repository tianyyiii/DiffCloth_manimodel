import sys
import os
sys.path.insert(0, os.path.abspath("./pylib"))
from jacobian import full_jacobian
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
import gc
import math
from renderer import WireframeRenderer
import pywavefront
import diffcloth_py as diffcloth

from datagen_framework import CONFIG, create_bent_curve_spacing, get_coord_by_idx, read_mesh_ignore_vtvn, set_sim_from_config, get_keypoints, step, render_record, create_bent_curve
from make_input_dataset import simplify_mesh


param_template_TS = {
    'name': "objs/DLG_Dress032_1.obj",
    'mesh_file': "src/assets/meshes/objs/DLG_Dress032_1.obj",
    'kp_file': "src/assets/meshes/objs/kp_DLG_Dress032_1.pcd",
    'drop_step': 150,
    'select_kp_idx': [[0,1], [2,3], [4, 5]],
    "line_points": 20,
    "bend_factor": 1.5,
    "point_spacing": 0.2,
}


TS_name = ['tshirt1000-tri']

def gen_param_by_name(name, subpath, param_template):
    param = param_template.copy()
    param['name'] = "objs/" + subpath + "/" + name + ".obj"
    param['mesh_file'] = "src/assets/meshes/objs/" + \
        subpath + "/" + name + ".obj"
    param['kp_file'] = "src/assets/meshes/objs/" + \
        subpath + "/kp_" + name + ".pcd"
    param['x0_file'] = "src/assets/meshes/objs/" + subpath + "/" + name + "_x0.npz"
    return param

def rotation_matrix_x_90():
    radians = np.radians(90)
    cos = np.cos(radians)
    sin = np.sin(radians)
    return torch.tensor(np.array([
        [1, 0, 0],
        [0, cos, -sin],
        [0, sin, cos]
    ]))


def get_nearby_point1(selected_idx: int, x0, distance_threshold=1.0, max_num=5):
    x0 = x0.reshape(-1, 3)
    p0 = x0[selected_idx]

    distances_to_p0 = torch.norm(x0 - p0, dim=1)
    near_p0_idx = torch.where(distances_to_p0 < distance_threshold)[0]
    near_p0_coords = x0[near_p0_idx]

    if near_p0_idx.shape[0] > max_num:
        near_p0_idx = near_p0_idx[:5]
        near_p0_coords = near_p0_coords[:5]

    return near_p0_idx.tolist(), near_p0_coords.detach().numpy()


def show(params):
    config = CONFIG.copy()
    config['fabric']['name'] = params['name']
    config['scene']['customAttachmentVertexIdx'] = [(0.0, [])]
    config['fabric']['k_stiff_stretching'] = 550
    config['fabric']['k_stiff_bending'] = 0.01
    config['fabric']['density'] = 1

    sim, x0, v0 = set_sim_from_config(config)
    kp_idx = [886, 391, 925, 827, 931, 1280]
    helper = diffcloth.makeOptimizeHelperWithSim("wear_hat", sim)
    pysim = pySim(sim, helper, True)

    mesh_faces = np.array(diffcloth.getSimMesh(sim))
    mesh_vertices = x0.detach().numpy().reshape(-1, 3)
    cloth_name = params["mesh_file"].split("/")[-1].split(".")[0]
    x0_path = os.path.join(os.path.dirname(params["mesh_file"]), f"{cloth_name}_x0.npz")

    # verts, faces = read_mesh_ignore_vtvn(params["mesh_file"])

    # select_kp_idx = [4, 7, 1, 8]
    select_kp_idxs = params["select_kp_idx"]

    # x0 = (x0.reshape(-1, 3) @ rotation_matrix_x_90().T).flatten()

    for i in tqdm.tqdm(range(params["drop_step"])):
        # stateInfo = sim.getStateInfo()
        # a = torch.tensor(a)
        a = torch.tensor([])
        x0, v0 = step(x0, v0, a, pysim)
        
        if i == 100:
            v0 = v0 * 0

    v0 = v0 * 0
    # save x0
    np.savez_compressed(x0_path, x0.detach().numpy())

    render_record(sim, kp_idx=kp_idx)

    for select_kp_idx in select_kp_idxs:
        near_p0_idx, near_p0_coords = get_nearby_point1(
                    kp_idx[select_kp_idx[0]], x0, distance_threshold=0.2)

        attach_point_list = near_p0_idx

        config['scene']['customAttachmentVertexIdx'] = [
            (0.0, attach_point_list)]
        sim, _, _ = set_sim_from_config(config)
        helper = diffcloth.makeOptimizeHelperWithSim("wear_hat", sim)
        pysim = pySim(sim, helper, True)

        p0 = get_coord_by_idx(x0, kp_idx[select_kp_idx[0]])
        p1 = get_coord_by_idx(x0, kp_idx[select_kp_idx[1]])
        # p2 = get_coord_by_idx(x0, kp_idx[select_kp_idx[2]])
        # p3 = get_coord_by_idx(x0, kp_idx[select_kp_idx[3]])

        num_points = 10
        line_points = params["line_points"]
        bend_factor = params["bend_factor"]
        point_spacing = params["point_spacing"]

        # curve_points1 = create_bent_curve(p0.detach().numpy(
        # ), p1.detach().numpy(), bend_factor=bend_factor, num_points=num_points)
        # curve_points2 = create_bent_curve(p2.detach().numpy(
        # ), p3.detach().numpy(), bend_factor=bend_factor, num_points=num_points)
        curve_pointsN = []
        for i in range(len(attach_point_list)):
            # curve_points1 = create_bent_curve_spacing(near_p0_coords[i]
            #                                           , p1.detach().numpy(), bend_factor=bend_factor, point_spacing=point_spacing)
            curve_points1 = create_bent_curve(near_p0_coords[i], p1.detach().numpy(), 
                                              bend_factor=bend_factor, num_points=num_points)
            curve_pointsN.append(curve_points1)
        curve_pointsN = np.array(curve_pointsN) #(attp_len, intp_len, 3)

        # curve_points1 = create_bent_curve_spacing(p0.detach().numpy(
        # ), p1.detach().numpy(), bend_factor=bend_factor, point_spacing=point_spacing)

        for i in tqdm.tqdm(range(curve_points1.shape[0])):

            p0_now = x0.reshape(-1, 3)[attach_point_list].detach().numpy()
            p0_interpolation = np.linspace(
                p0_now, curve_pointsN[:, i, :], line_points)

            for j in range(line_points):
                a = torch.tensor(p0_interpolation[j].flatten())
                x0, v0 = step(x0, v0, a, pysim)
                v0[kp_idx[select_kp_idx[0]] *
                    3:(kp_idx[select_kp_idx[0]]+1)*3] = 0

            # v0 = v0 * 0
        
        for _ in range(20):
            a = torch.tensor(
                    curve_pointsN[:, -1, :].flatten())
            x0, v0 = step(x0, v0, a, pysim)

        render_record(sim, [kp_idx[select_kp_idx[0]], kp_idx[select_kp_idx[1]]
                            ], curves=[curve_points1])



if __name__ == '__main__':

    # simplify_mesh('src/assets/meshes/objs/TS/TNSC_Tshirt_Ts1_0.obj', 'src/assets/meshes/objs/TS/TNSC_Tshirt_Ts1_0s.obj')

    for name in TS_name:
        param = gen_param_by_name(name, "TS", param_template_TS)
        show(param)