import re
import matplotlib
from matplotlib import pyplot as plt
from datagen_framework import CONFIG, read_mesh_ignore_vtvn, render_record, set_sim_from_config
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
# set matplotlib backend web
matplotlib.use('webagg')


def step(x, v, a, simModule):
    x1, v1 = simModule(x, v, a)
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
                dist_mat = np.linalg.norm(
                    init_state - init_state[max_point], axis=1)
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

                diff = np.average(
                    np.sqrt(np.linalg.norm(gt_x - pred_x, axis=1)))
                diffs_tmp.append(diff)

            # plt.plot(diffs_tmp)
            # plt.show()

            diffs.append(diff)

            np.save(out_path + f'{cloth_name}_random_{i}_{j}_gt.npy', gt_x)
            np.save(out_path + f'{cloth_name}_random_{i}_{j}_pred.npy', x)
            # np.save(out_path + f'{cloth_name}_random_{i}_{j}_gt_atp.npy', diff)
            # np.save(out_path + f'{cloth_name}_random_{i}_{j}_pred_atp.npy', x.reshape(-1, 3).detach().numpy()[idx_map[att_p]])
            np.savez(out_path + f'{cloth_name}_random_{i}_{j}_heatmap', init_state=init_state,
                     pred_atp=init_state[att_p], gt_atp=init_state[input_data["attached_point"][i//20, i % 20]], pred_hm=hm, gt_hm=data_heatmap["heatmap"][i//20, i % 20])

            pass

        all_diff.append(diffs)

    print(np.array(all_diff))

    pass


def norm_pointcloud(point_cloud):
    point_cloud = point_cloud - np.mean(point_cloud, axis=0)
    point_cloud = point_cloud / np.max(np.abs(point_cloud))
    return point_cloud


def evaluate_new(data_path):
    if data_path[-1] != '/':
        data_path += '/'

    # cloth_obj_paths = [
    #     "src/assets/meshes/objs/Long_LongSleeve",
    #     "src/assets/meshes/objs/Long_NoSleeve",
    # ]

    # find the pkl file in data_path
    pkl_files = [f for f in os.listdir(data_path) if f.endswith(".pkl")]
    if len(pkl_files) == 0:
        print("No pkl file found in data_path")
        return
    pkl_file = pkl_files[0]

    input_data = np.load(data_path + pkl_file, allow_pickle=True)

    pred_xs = []
    gt_xs = []
    all_diffs = []
    all_inits = []
    cloth_names = []

    for i, cloth_name in tqdm.tqdm(enumerate(input_data["cloth_name"])):
        # DLNS_Dress001_random_0_4800_heatmap.npz
        real_cloth_name = cloth_name[:cloth_name.find("_random_")]
        cloth_name_random = cloth_name[:cloth_name.find("_random_") + 9]
        cloth_name_random_original = cloth_name[:cloth_name.find(
            "_random_") + 7]

        obj_path = data_path + real_cloth_name + ".obj"
        random_data_path = data_path + cloth_name_random + ".npz"

        random_data = np.load(random_data_path, allow_pickle=True)
        original_data = np.load(
            data_path + cloth_name_random_original + ".npz", allow_pickle=True)

        config = CONFIG.copy()

        config["fabric"]["name"] = obj_path[obj_path.find("meshes") + 7:]
        config["fabric"]["k_stiff_stretching"] = random_data["k_stiff_stretching"].item()
        config["fabric"]["k_stiff_bending"] = random_data["k_stiff_bending"].item()
        config["fabric"]["density"] = random_data["density"].item(
        ) if "density" in random_data.keys() else 1

        point_cloud = input_data["point_cloud"][i]
        pred_contact_points_heatmap = input_data["pred_contact_points_heatmap"][i]
        pred_contact_force_map = input_data["pred_contact_force_map"][i]
        gt_contact_point_id = input_data["gt_contact_point_id"][i]

        sample_idx = random_data["sample_idx"]

        data_idx = -1
        data_idx_m = -1
        norm_coeff = 1

        gt_contact_point_id_expand = gt_contact_point_id[:2].reshape(1, 1, 2)
        matches = np.all(random_data["attached_point"]
                         == gt_contact_point_id_expand, axis=-1)
        where = np.where(matches)
        if len(where[0]) == 0:
            gt_contact_point_id_expand = gt_contact_point_id[:2][::-1].reshape(
                1, 1, 2)
            matches = np.all(
                random_data["attached_point"] == gt_contact_point_id_expand, axis=-1)
            where = np.where(matches)

        if len(where[0]) > 0 and len(where[1]) > 0:
            data_idx = where[0][0]
            data_idx_m = where[1][0]
            norm_coeff = np.max(random_data["init_state"][data_idx])

        # for j in range(random_data["init_state"].shape[0]):
        #     where = np.where(
        #         random_data["attached_point"][j] == gt_contact_point_id[:2])
        #     # assume 2 attachpoints
        #     if len(where[0]) > 1 and len(where[1]) > 1 and where[0][0] == where[0][1]:
        #         if np.allclose(norm_pointcloud(random_data["init_state"][j]), point_cloud):
        #             data_idx = j
        #             data_idx_m = where[0][0]
        #             norm_coeff = np.max(random_data["init_state"][j])
        #             break
        #         else:
        #             print("point cloud not match")

        if data_idx == -1:
            print(f"Cannot find the data for {cloth_name}")
            continue

        all_init_state = original_data["data"][data_idx]["init_state"]

        hm = pred_contact_points_heatmap
        force = pred_contact_force_map

        # get the attachment point
        # get the max point index from heatmap, then set all points < radius to 0, loop until no points > 2
        radius = 0.1
        max_pred_value = hm.max()
        att_p = []

        while not np.all(hm < max_pred_value / 2):
            # set all points < radius to 0
            max_point = np.where(hm == hm.max())[0]
            att_p.append(max_point.item())
            dist_mat = np.linalg.norm(
                point_cloud - point_cloud[max_point], axis=1)
            hm[dist_mat < radius] = 0
            pass

        config['scene']['customAttachmentVertexIdx'] = [
            (0.0, sample_idx[att_p])]
        sim, x, v = set_sim_from_config(config)
        helper = diffcloth.makeOptimizeHelperWithSim("wear_hat", sim)
        pysim = pySim(sim, helper, True)

        x = torch.tensor(all_init_state).flatten().to(torch.float64)
        v = torch.zeros_like(x)

        action_att_p = force[att_p] * norm_coeff * 2

        diffs_tmp = []
        for _ in range(10):
            act = torch.tensor(action_att_p).to(
                torch.float64).flatten() + x.reshape(-1, 3)[sample_idx[att_p]].flatten()
            x, v = step(x, v, act, pysim)

            gt_x = original_data["data"][data_idx]["all_target_state"][data_idx_m]
            pred_x = x.reshape(-1, 3).detach().numpy()

            diff = np.average(
                np.sqrt(np.linalg.norm(gt_x - pred_x, axis=1)))
            diffs_tmp.append(diff)

        # plt.plot(diffs_tmp)
        # plt.show()

        all_diffs.append(diffs_tmp)
        pred_xs.append(pred_x)
        gt_xs.append(gt_x)
        all_inits.append(all_init_state[sample_idx])
        cloth_names.append(cloth_name)

        print(f"{cloth_name}: {np.min(diffs_tmp)}")

        # render_record(sim, sample_idx[att_p])

    # save
    np.savez_compressed(data_path + "evaluate_result",
                        pred_xs=pred_xs, gt_xs=gt_xs, all_diffs=all_diffs, all_inits=all_inits, cloth_names=cloth_names)
    pass


def extract_path_info_from_cloth_name_tt(cloth_name):
    # find the number_number in cloth_name using regex \d+_\d+
    match = re.search(r'\d+_\d+', cloth_name)
    if match is None:
        print(f"Cannot find the number_number in {cloth_name}")
        return None

    cloth_clazz_str = cloth_name[:match.start() - 1]
    cloth_clazz = cloth_clazz_str[:cloth_clazz_str.find("_")]
    cloth_category = cloth_clazz_str[cloth_clazz_str.find("_") + 1:]
    cloth_number_str = match.group()
    cloth_number = int(cloth_number_str[:cloth_number_str.find("_")])

    return cloth_clazz, cloth_category, cloth_number, cloth_number_str


def calculate_sample_idxs(points, mesh_points):
    original_points = points[:mesh_points]
    sampled_points = points[mesh_points:]

    sampled_indices = []

    for point in sampled_points:
        index = np.where(original_points == point)[0][0]
        sampled_indices.append(index)

    sampled_indices = np.concatenate(
        (np.arange(mesh_points), np.array(sampled_indices)))
    return sampled_indices


def evaluate_tie(input_data, i):
    # TODO implement this

    obj_path = 'src/assets/meshes/evaluate/TT/tie/tie.obj'
    # ensure obj_path exists
    if not os.path.exists(obj_path):
        raise ValueError(f"Cannot find {obj_path}")

    random_data_path = 'src/assets/meshes/evaluate/TT/tie/tie_2223_heatmap.npz'
    # ensure random_data_path exists
    if not os.path.exists(random_data_path):
        raise ValueError(f"Cannot find {random_data_path}")

    random_data = np.load(random_data_path, allow_pickle=True)

    config = CONFIG.copy()
    config["fabric"]["name"] = obj_path[obj_path.find("meshes") + 7:]
    config["fabric"]["k_stiff_stretching"] = random_data["k_stiff_stretching"].item()
    config["fabric"]["k_stiff_bending"] = random_data["k_stiff_bending"].item()
    config["fabric"]["density"] = random_data["density"].item(
    ) if "density" in random_data.keys() else 1

    point_cloud = input_data["point_cloud"][i]
    pred_contact_points_heatmap = input_data["pred_contact_points_heatmap"][i]
    pred_contact_force_map = input_data["pred_contact_force_map"][i]
    gt_contact_point_id = input_data["gt_contact_point_id"][i]

    data_idx = -1
    data_idx_m = -1
    norm_coeff = 1

    for j in range(random_data["init_state"].shape[0]):
        if np.allclose(norm_pointcloud(random_data["init_state"][j]), norm_pointcloud(point_cloud)):
            data_idx = j
            data_idx_m = -1
            norm_coeff = np.max(random_data["init_state"][j])
            break

    gt_contact_point_id_expand = gt_contact_point_id.reshape(1, 1, 4)
    matches = np.all(random_data["attached_point"]
                     == gt_contact_point_id_expand, axis=-1)
    where = np.where(matches)
    if len(where[0]) == 0:
        gt_contact_point_id_expand = gt_contact_point_id[:2][::-1].reshape(
            1, 1, 2)
        matches = np.all(
            np.round(random_data["attached_point"]).astype(int) == gt_contact_point_id_expand, axis=-1)
        where = np.where(matches)

    pass


def evaluate_tt_class(data_path):
    if data_path[-1] != '/':
        data_path += '/'

    # find the pkl
    pkl_files = [f for f in os.listdir(data_path) if f.endswith(".pkl")]
    if len(pkl_files) == 0:
        print("No pkl file found in data_path")
        return
    pkl_file = pkl_files[0]

    input_data = np.load(data_path + pkl_file, allow_pickle=True)

    pred_xs = []
    gt_xs = []
    all_diffs = []
    all_inits = []
    cloth_names = []

    for i, cloth_name in tqdm.tqdm(enumerate(input_data["cloth_name"])):
        # find the last / in cloth_name
        cloth_name = cloth_name[cloth_name.rfind("/") + 1:]

        if cloth_name.startswith("tie"):
            evaluate_tie(input_data, i)
            # TODO Save the result here
            continue
            pass

        inf = extract_path_info_from_cloth_name_tt(cloth_name)
        if inf is None:
            raise ValueError(f"Cannot extract path info from {cloth_name}")
        cloth_clazz, cloth_category, cloth_number, cloth_number_str = inf

        obj_path = data_path + 'objs/' + cloth_clazz + '/' + \
            cloth_category + '/' + str(cloth_number) + '/mesh.obj'
        # ensure obj_path exists
        if not os.path.exists(obj_path):
            raise ValueError(f"Cannot find {obj_path}")

        mesh_vertices_number = read_mesh_ignore_vtvn(obj_path)[0].shape[0]

        random_data_path = data_path + 'random_data/' + cloth_clazz + '/' + \
            cloth_category + '/' + cloth_number_str + '/cloth_resampled.npz'
        # random_data_path = data_path + 'random_data/' + cloth_name
        # ensure random_data_path exists
        if not os.path.exists(random_data_path):
            # create parent folder
            os.makedirs(os.path.dirname(random_data_path), exist_ok=True)
            # try to download from remote server
            url = 'http://36.212.171.219:21333/' + cloth_clazz + '/' + \
                cloth_category + '/' + cloth_number_str + '/cloth_resampled.npz'
            os.system(f"wget {url} -O {random_data_path}")
            if not os.path.exists(random_data_path):
                raise ValueError(f"Cannot find {random_data_path}")

        random_data = np.load(random_data_path, allow_pickle=True)

        config = CONFIG.copy()
        config["fabric"]["name"] = obj_path[obj_path.find("meshes") + 7:]
        config["fabric"]["k_stiff_stretching"] = random_data["kp"].item()
        config["fabric"]["k_stiff_bending"] = random_data["kd"].item()
        config["fabric"]["density"] = random_data["density"].item(
        ) if "density" in random_data.keys() else 1

        point_cloud = input_data["point_cloud"][i]
        pred_contact_points_heatmap = input_data["pred_contact_points_heatmap"][i]
        pred_contact_force_map = input_data["pred_contact_force_map"][i]
        gt_contact_point_id = input_data["gt_contact_point_id"][i]

        data_idx = -1
        data_idx_m = -1
        norm_coeff = 1

        gt_contact_point_id_expand = gt_contact_point_id.reshape(1, 1, 4)
        matches = np.all(random_data["attached_point"]
                         == gt_contact_point_id_expand, axis=-1)
        where = np.where(matches)
        if len(where[0]) == 0:
            gt_contact_point_id_expand[0, 0, 0], gt_contact_point_id_expand[0, 0,
                                                                            1] = gt_contact_point_id_expand[0, 0, 1], gt_contact_point_id_expand[0, 0, 0]
            matches = np.all(random_data["attached_point"]
                             == gt_contact_point_id_expand, axis=-1)
            where = np.where(matches)

        if len(where[0]) > 0 and len(where[1]) > 0:
            data_idx = where[0][0]
            data_idx_m = where[1][0]
            norm_coeff = np.max(random_data["init_state"][data_idx])

        if data_idx == -1:
            print(f"Cannot find the data for {cloth_name}")
            raise ValueError(f"Cannot find the data for {cloth_name}")
            continue

        hm = pred_contact_points_heatmap
        force = pred_contact_force_map

        # get the attachment point
        # get the max point index from heatmap, then set all points < radius to 0, loop until no points > 2
        radius = 0.1
        max_pred_value = hm.max()
        att_p = []

        while not np.all(hm < max_pred_value / 2):
            # set all points < radius to 0
            max_point = np.where(hm == hm.max())[0]
            att_p.append(max_point.item())
            dist_mat = np.linalg.norm(
                point_cloud - point_cloud[max_point], axis=1)
            hm[dist_mat < radius] = 0
            pass

        all_init_state = random_data["init_state"][data_idx][:mesh_vertices_number, :]
        # construct sample_idx from all_init_state
        sample_idx = calculate_sample_idxs(
            random_data["init_state"][data_idx], mesh_vertices_number)

        config['scene']['customAttachmentVertexIdx'] = [
            (0.0, sample_idx[att_p])]
        sim, x, v = set_sim_from_config(config)
        helper = diffcloth.makeOptimizeHelperWithSim("wear_hat", sim)
        pysim = pySim(sim, helper, True)

        x = torch.tensor(all_init_state).flatten().to(torch.float64)
        v = torch.zeros_like(x)

        action_att_p = force[att_p] * norm_coeff

        diffs_tmp = []
        for _ in range(10):
            act = torch.tensor(action_att_p).to(
                torch.float64).flatten() + x.reshape(-1, 3)[sample_idx[att_p]].flatten()
            x, v = step(x, v, act, pysim)

            gt_x = random_data["target_state"][data_idx,
                                               data_idx_m][:mesh_vertices_number, :]
            pred_x = x.reshape(-1, 3).detach().numpy()

            diff = np.average(
                np.sqrt(np.linalg.norm(gt_x - pred_x, axis=1)))
            diffs_tmp.append(diff)

        # render_record(sim, sample_idx[att_p])
        plt.plot(diffs_tmp)
        if i % 10 == 0 and i > 0:
            plt.show()
            # print the min diff index of all the diffs
            print(np.argmin(np.array(all_diffs), axis=1))

        all_diffs.append(diffs_tmp)
        pred_xs.append(pred_x)
        gt_xs.append(gt_x)
        all_inits.append(all_init_state[sample_idx])
        cloth_names.append(cloth_name)

        print(f"{cloth_name}: {np.min(diffs_tmp)}")

    # save result
    # np.savez_compressed(data_path + "evaluate_result",
    #                     pred_xs=pred_xs, gt_xs=gt_xs, all_diffs=all_diffs, all_inits=all_inits)
    np.savez_compressed(data_path + "evaluate_result",
                        pred_xs=np.array(pred_xs, dtype=object), gt_xs=np.array(gt_xs, dtype=object), all_diffs=np.array(all_diffs, dtype=object), all_inits=np.array(all_inits, dtype=object), cloth_names=cloth_names)
    pass


if __name__ == '__main__':
    # evaluate("src/assets/meshes/evaluate/DSG_Dress102")
    # evaluate_new("src/assets/meshes/evaluate/test")
    evaluate_tt_class("src/assets/meshes/evaluate/TT")
