from matplotlib import pyplot as plt
from datagen_framework import CONFIG, read_mesh_ignore_vtvn, set_sim_from_config
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
import sys
import os
sys.path.insert(0, os.path.abspath("./pylib"))
import matplotlib
# set matplotlib backend web
matplotlib.use('webagg')


def stepF(x, v, a, f, simModule):
    x1, v1 = simModule(x, v, a, f)
    return x1, v1


def evaluate(data_path, out_path="./output/"):
    # ensure data_path ends with '/'
    if data_path[-1] != '/':
        data_path += '/'

    # get the cloth name
    cloth_name = data_path.split('/')[-2]

    contact_force = np.load(data_path + 'contact_force.npy')
    contact_heatmap = np.load(data_path + 'contact_heatmap.npy')

    data_heatmap = np.load(data_path + f'{cloth_name}_random_0_heatmap.npz')
    input_data = np.load(data_path + f'{cloth_name}_random_0.npz')
    origin_data = np.load(
        data_path + f'{cloth_name}_random.npz', allow_pickle=True)

    # get the idx mapping from input_data to origin_data
    x0 = origin_data["data"][0]["init_state"]
    x1 = input_data["init_state"][0]
    # find the same vertices in x0 and x1
    idx_map = np.array([np.where(np.all(x0 == x1[i, :], axis=1))[
                       0][0] for i in range(len(x1))])

    config = CONFIG.copy()
    config["fabric"]["name"] = "evaluate/" + \
        cloth_name + "/" + cloth_name + ".obj"
    config["fabric"]["k_stiff_stretching"] = input_data["k_stiff_stretching"].item()
    config["fabric"]["k_stiff_bending"] = input_data["k_stiff_bending"].item()
    config["fabric"]["density"] = input_data["density"].item(
    ) if "density" in input_data.keys() else 1
    
    all_diff = []

    for i in range(contact_heatmap.shape[0]):
        
        diffs = []
        
        for j in range(contact_heatmap.shape[1]):

            hm = contact_heatmap[i, j]
            force = contact_force[i, j]

            # get the attachment point
            # get the max point index from heatmap, then set all points < radius to 0, loop until no points > 2
            radius = 0.1
            max_pred_value = hm.max()
            att_p = []
            init_state = input_data["init_state"][i // 20]
            
            
            while not np.all(hm < max_pred_value / 2):
                # set all points < radius to 0
                max_point = np.where(hm == hm.max())[0]
                att_p.append(max_point.item())
                dist_mat = np.linalg.norm(init_state - init_state[max_point], axis=1)
                hm[dist_mat < radius] = 0
                pass
            
            # att_p = list(np.where(hm > 0.8)[0])
            
            
            
            config['scene']['customAttachmentVertexIdx'] = [
                (0.0, [])]

            sim, x, v = set_sim_from_config(config)
            helper = diffcloth.makeOptimizeHelperWithSim("wear_hat", sim)
            pysim = pySimF(sim, helper, True)

            x = torch.tensor(origin_data["data"]
                             [i // 20]["init_state"]).flatten()
            v = torch.zeros_like(x)
            # f = torch.ones_like(x).reshape(-1, 3) * -9.8
            f = torch.zeros_like(x).reshape(-1, 3)
            f[idx_map[att_p]] = torch.tensor(force)[att_p] * 100
            f = f.flatten()
            
            diffs_tmp = []
            for _ in range(45):
                x, v = stepF(x, v, torch.tensor([]), f, pysim)

                gt_x = input_data["all_target_state"][i // 20, i % 20]
                pred_x = x.reshape(-1, 3).detach().numpy()[idx_map]

                diff = np.average(np.sqrt(np.linalg.norm(gt_x - pred_x, axis=1)))
                diffs_tmp.append(diff)
                
            # plt.plot(diffs_tmp)
            # plt.show()
            
            diffs.append(diff)

            np.save(out_path + f'{cloth_name}_random_{i}_{j}_gt.npy', gt_x)
            np.save(out_path + f'{cloth_name}_random_{i}_{j}_pred.npy', x)
            # np.save(out_path + f'{cloth_name}_random_{i}_{j}_gt_atp.npy', diff)
            # np.save(out_path + f'{cloth_name}_random_{i}_{j}_pred_atp.npy', x.reshape(-1, 3).detach().numpy()[idx_map[att_p]])
            np.savez(out_path + f'{cloth_name}_random_{i}_{j}_heatmap', init_state=init_state, 
                    pred_atp=init_state[att_p], gt_atp=init_state[input_data["attached_point"][i//20, i%20]]
                    ,pred_hm=hm, gt_hm=data_heatmap["heatmap"][i//20, i%20])
            
            pass
        
        all_diff.append(diffs)

    print(np.array(all_diff))

    pass


if __name__ == '__main__':
    evaluate("src/assets/meshes/evaluate/DSG_Dress102")
