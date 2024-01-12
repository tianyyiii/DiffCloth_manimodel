import sys
import os

from datagen_framework import CONFIG, set_sim_from_config, step
sys.path.insert(0, os.path.abspath("./pylib"))
import diffcloth_py as diffcloth
import pywavefront
from renderer import WireframeRenderer
import math
import gc
from pySim.pySim import pySim, pySimF
from pySim.functional import SimFunction
import common
import tqdm
import numpy as np
import torch
import trimesh
import open3d as o3d
import random
import time
import json
import io
import contextlib
from jacobian import full_jacobian

def pre_drop(one_class_path, class_name):
    # get all objs in path
    objs = []
    for root, dirs, files in os.walk(one_class_path):
        for file in files:
            if file.endswith(".obj"):
                objs.append(os.path.join(root, file))
    
    for obj in objs:
        cloth_name = obj.split("/")[-1].split(".")[0]
        
        config = CONFIG.copy()
        config['fabric']['name'] = "objs/" + class_name + "/" + cloth_name + ".obj"
        config['scene']['customAttachmentVertexIdx'] = [(0.0, [])]
        sim, x0, v0 = set_sim_from_config(config)
        helper = diffcloth.makeOptimizeHelperWithSim("wear_hat", sim)
        pysim = pySim(sim, helper, True)

        for i in tqdm.tqdm(range(100)):
            # stateInfo = sim.getStateInfo()
            # a = torch.tensor(a)
            a = torch.tensor([])
            x0, v0 = step(x0, v0, a, pysim)

        np.savez_compressed(one_class_path + "/" + cloth_name + "_x0" + ".npz", x0.detach().numpy())
    

def write_mesh(path, mesh):
    # ignore all normals and texture coordinates
    mesh = trimesh.Trimesh(
        vertices=mesh.vertices,
        faces=mesh.triangles,
        process=False,
    )
    mesh.remove_degenerate_faces()
    mesh.remove_duplicate_faces()
    mesh.remove_infinite_values()
    mesh.remove_unreferenced_vertices()
    mesh.export(path, file_type="obj")


def simplify_mesh(mesh_file, out_file):
    mesh = o3d.io.read_triangle_mesh(mesh_file)

    vertices_num = np.asarray(mesh.vertices).shape[0]
    if vertices_num < 3500:
        print("vertices_num < target_vertices")
        write_mesh(out_file, mesh)
        return
    
    # target_triangles = int(3500 * 1.5)
    # simplified_mesh = mesh.simplify_quadric_decimation(target_triangles)
    simplified_mesh = mesh
    for _ in range(1000):
        tri_num = np.asarray(simplified_mesh.triangles).shape[0]
        tri_num = int(tri_num * 0.9)
        simplified_mesh = mesh.simplify_quadric_decimation(tri_num)
        vertices_num = np.asarray(simplified_mesh.vertices).shape[0]
        if vertices_num < 3500:
            break

    # try_voxel_size = np.linspace(0.01, 5, 500)
    # for voxel_size in try_voxel_size:
    #     simplified_mesh = mesh.simplify_vertex_clustering(voxel_size=voxel_size, contraction=o3d.geometry.SimplificationContraction.Quadric)
    #     vertices_num = np.asarray(simplified_mesh.vertices).shape[0]
    #     if vertices_num < 3500:
    #         break
    simplified_mesh.remove_non_manifold_edges()
    write_mesh(out_file, simplified_mesh)
    

def pre_simplify_mesh(one_class_path):
    # get all objs in path
    objs = []
    for root, dirs, files in os.walk(one_class_path):
        for file in files:
            if file.endswith(".obj"):
                objs.append(os.path.join(root, file))
    
    for obj in tqdm.tqdm(objs):
        simplify_mesh(obj, obj)


if __name__ == '__main__':
    # pre_drop("src/assets/meshes/objs/Long_NoSleeve", "Long_NoSleeve")
    # pre_drop("src/assets/meshes/objs/Long_ShortSleeve", "Long_ShortSleeve")
    # pre_drop("src/assets/meshes/objs/Long_Tube", "Long_Tube")
    # pre_drop("src/assets/meshes/objs/Short_Gallus", "Short_Gallus")
    # pre_drop("src/assets/meshes/objs/Short_NoSleeve", "Short_NoSleeve")
    # pre_drop("src/assets/meshes/objs/Short_ShortSleeve", "Short_ShortSleeve")
    # pre_drop("src/assets/meshes/objs/Short_Tube", "Short_Tube")

    # pre_simplify_mesh("src/assets/meshes/objs/Long_LongSleeve")
    # pre_simplify_mesh("src/assets/meshes/objs/Long_NoSleeve")
    # pre_simplify_mesh("src/assets/meshes/objs/Long_ShortSleeve")
    # pre_simplify_mesh("src/assets/meshes/objs/Long_Tube")
    # pre_simplify_mesh("src/assets/meshes/objs/Short_Gallus")
    # pre_simplify_mesh("src/assets/meshes/objs/Short_NoSleeve")
    # pre_simplify_mesh("src/assets/meshes/objs/Short_ShortSleeve")
    # pre_simplify_mesh("src/assets/meshes/objs/Short_Tube")

    # pre_drop("src/assets/meshes/objs/Long_LongSleeve", "Long_LongSleeve")
    # pre_drop("src/assets/meshes/objs/Long_NoSleeve", "Long_NoSleeve")
    # pre_drop("src/assets/meshes/objs/Long_ShortSleeve", "Long_ShortSleeve")
    # pre_drop("src/assets/meshes/objs/Long_Tube", "Long_Tube")
    # pre_drop("src/assets/meshes/objs/Short_Gallus", "Short_Gallus")
    # pre_drop("src/assets/meshes/objs/Short_NoSleeve", "Short_NoSleeve")
    # pre_drop("src/assets/meshes/objs/Short_ShortSleeve", "Short_ShortSleeve")
    # pre_drop("src/assets/meshes/objs/Short_Tube", "Short_Tube")

    pre_simplify_mesh("src/assets/meshes/objs/Short_LongSleeve")
    pre_drop("src/assets/meshes/objs/Short_LongSleeve", "Short_LongSleeve")

    # params = {
    #     'name': "objs/DLLS_Dress008_0.obj",
    #     'mesh_file': "src/assets/meshes/objs/DLLS_Dress008_0.obj",
    #     'kp_file': "src/assets/meshes/objs/kp_DLLS_Dress008_0.pcd",
    #     'drop_step': 100,
    #     'select_kp_idx': [[2, 8, 0, 7], [4, 7, 1, 8]],
    #     "line_points": 20,
    #     "bend_factor": 1.5,
    #     "point_spacing": 0.2,
    # }
    
    pass