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
    

if __name__ == '__main__':
    pre_drop("src/assets/meshes/objs/Long_LongSleeve", "Long_LongSleeve")
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