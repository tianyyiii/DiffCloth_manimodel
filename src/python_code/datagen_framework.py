import sys
import os
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


def create_bent_curve_spacing(p0, p3, bend_factor=0.5, point_spacing=None):
    """Create a curve that bends towards the z-axis and passes through p0 and p3.

    If point_spacing is provided, points are placed at approximately this spacing along the curve."""
    # Calculate control points for bending
    p1 = p0 + np.array([0, bend_factor, 0])
    p2 = p3 + np.array([0, bend_factor, 0])

    curve_points = [p0]
    if point_spacing is not None:
        t = 0
        while t < 1:
            # Find the next t where the distance is approximately equal to point_spacing
            t_next = t
            distance = 0
            while distance < point_spacing and t_next < 1:
                # Increment t_next. The step size can be adjusted for accuracy.
                t_next += 0.01
                next_point = cubic_bezier(p0, p1, p2, p3, t_next)
                distance = np.linalg.norm(next_point - curve_points[-1])

            if t_next < 1:
                curve_points.append(next_point)
            t = t_next
    else:
        num_points = 100  # Default number of points
        t_values = np.linspace(0, 1, num_points)
        curve_points = [cubic_bezier(p0, p1, p2, p3, t) for t in t_values]

    return np.array(curve_points)


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


def calculate_vertex_normal(v, f):
    mesh = trimesh.Trimesh(vertices=v, faces=f)
    vertex_normals = mesh.vertex_normals
    return vertex_normals


def task(params):
    config = CONFIG.copy()
    config['fabric']['name'] = params['name']
    config['scene']['customAttachmentVertexIdx'] = [(0.0, [])]
    sim, x0, v0 = set_sim_from_config(config)
    kp_idx = get_keypoints(params["mesh_file"],
                           params["kp_file"])
    helper = diffcloth.makeOptimizeHelperWithSim("wear_hat", sim)
    pysim = pySim(sim, helper, True)

    data = []
    mesh_faces = np.array(diffcloth.getSimMesh(sim))
    mesh_vertices = x0.detach().numpy().reshape(-1, 3)

    select_kp_idxs = params["select_kp_idx"]

    # for i in tqdm.tqdm(range(params["drop_step"])):
    #     # stateInfo = sim.getStateInfo()
    #     # a = torch.tensor(a)
    #     a = torch.tensor([])
    #     x0, v0 = step(x0, v0, a, pysim)
    x0 = torch.tensor(np.load(params["x0_file"])["arr_0"])
    
    v0 = v0 * 0
    # render_record(sim)

    for select_kp_idx in select_kp_idxs:
        config['scene']['customAttachmentVertexIdx'] = [
            (0.0, [kp_idx[select_kp_idx[0]], kp_idx[select_kp_idx[2]]])]
        sim, _, _ = set_sim_from_config(config)
        helper = diffcloth.makeOptimizeHelperWithSim("wear_hat", sim)
        pysim = pySim(sim, helper, True)

        p0 = get_coord_by_idx(x0, kp_idx[select_kp_idx[0]])
        p1 = get_coord_by_idx(x0, kp_idx[select_kp_idx[1]])
        p2 = get_coord_by_idx(x0, kp_idx[select_kp_idx[2]])
        p3 = get_coord_by_idx(x0, kp_idx[select_kp_idx[3]])

        num_points = 5
        line_points = params["line_points"]
        bend_factor = params["bend_factor"]
        point_spacing = params["point_spacing"]

        # curve_points1 = create_bent_curve(p0.detach().numpy(
        # ), p1.detach().numpy(), bend_factor=bend_factor, num_points=num_points)
        # curve_points2 = create_bent_curve(p2.detach().numpy(
        # ), p3.detach().numpy(), bend_factor=bend_factor, num_points=num_points)

        curve_points1 = create_bent_curve_spacing(p0.detach().numpy(
        ), p1.detach().numpy(), bend_factor=bend_factor, point_spacing=point_spacing)
        curve_points2 = create_bent_curve_spacing(p2.detach().numpy(
        ), p3.detach().numpy(), bend_factor=bend_factor, point_spacing=point_spacing)

        if curve_points1.shape[0] != curve_points2.shape[0]:
            max_points_num = max(
                curve_points1.shape[0], curve_points2.shape[0])
            curve_points1 = create_bent_curve(p0.detach().numpy(), p1.detach(
            ).numpy(), bend_factor=bend_factor, num_points=max_points_num)
            curve_points2 = create_bent_curve(p2.detach().numpy(), p3.detach(
            ).numpy(), bend_factor=bend_factor, num_points=max_points_num)
            num_points = max_points_num
        else:
            num_points = curve_points1.shape[0]

        # remove first points
        curve_points1 = curve_points1[1:]
        curve_points2 = curve_points2[1:]
        num_points -= 1

        for i in tqdm.tqdm(range(num_points)):
            data_i = {}

            p0_now = get_coord_by_idx(
                x0, kp_idx[select_kp_idx[0]]).detach().numpy()
            p2_now = get_coord_by_idx(
                x0, kp_idx[select_kp_idx[2]]).detach().numpy()
            p0_interpolation = np.linspace(
                p0_now, curve_points1[i], line_points + 1)[1:]
            p2_interpolation = np.linspace(
                p2_now, curve_points2[i], line_points + 1)[1:]

            data_i["init_state"] = x0.detach().numpy().reshape(-1, 3)
            data_i["init_state_normal"] = calculate_vertex_normal(
                data_i["init_state"], mesh_faces)

            data_i["attached_point"] = np.array(
                [kp_idx[select_kp_idx[0]], kp_idx[select_kp_idx[2]]] * line_points).reshape(-1, 2)
            data_i["attached_point_target"] = np.stack(
                (p0_interpolation, p2_interpolation), axis=1)

            data_i["target_state"] = []
            data_i["target_state_normal"] = []

            for j in range(line_points):
                a = torch.tensor(np.concatenate(
                    (p0_interpolation[j], p2_interpolation[j])))
                x0, v0 = step(x0, v0, a, pysim)
                # v0[kp_idx[select_kp_idx[0]]*3:(kp_idx[select_kp_idx[0]]+1)*3] = 0
                # v0[kp_idx[select_kp_idx[2]]*3:(kp_idx[select_kp_idx[2]]+1)*3] = 0
                all_target_state = x0.detach().numpy().reshape(-1, 3)
                all_target_state_normal = calculate_vertex_normal(
                    all_target_state, mesh_faces)
                data_i["target_state"].append(all_target_state[kp_idx])
                data_i["target_state_normal"].append(
                    all_target_state_normal[kp_idx])

            data_i["target_state"] = np.stack(data_i["target_state"], axis=0)
            data_i["target_state_normal"] = np.stack(
                data_i["target_state_normal"], axis=0)
            v0 = v0 * 0

            # data_i["target_state"] = x0.detach().numpy().reshape(-1, 3)
            # data_i["target_state_normal"] = calculate_vertex_normal(
            #     data_i["target_state"], mesh_faces)
            # save

            jacobian = full_jacobian(
                mesh_vertices, mesh_faces, x0, v0, kp_idx, config)
            data_i["response_matrix"] = jacobian

            # jacobian = calculate_jacobian(x0, v0, config, kp_idx)
            # data_i["response_matrix"] = jacobian.detach().numpy()
            # print(jacobian[0].shape)

            data.append(data_i)

            # break # only one step for debug

    np.savez_compressed("unprocessed.npz", data=data)

    frictional_coeff = 0.5
    return data, kp_idx, frictional_coeff, config["fabric"]["k_stiff_stretching"], config["fabric"]["k_stiff_bending"]


def show(params):
    config = CONFIG.copy()
    config['fabric']['name'] = params['name']
    config['scene']['customAttachmentVertexIdx'] = [(0.0, [])]
    sim, x0, v0 = set_sim_from_config(config)
    kp_idx = get_keypoints(params["mesh_file"],
                           params["kp_file"])
    helper = diffcloth.makeOptimizeHelperWithSim("wear_hat", sim)
    pysim = pySim(sim, helper, True)

    mesh_faces = np.array(diffcloth.getSimMesh(sim))
    mesh_vertices = x0.detach().numpy().reshape(-1, 3)

    # select_kp_idx = [4, 7, 1, 8]
    select_kp_idxs = params["select_kp_idx"]

    for i in tqdm.tqdm(range(params["drop_step"])):
        # stateInfo = sim.getStateInfo()
        # a = torch.tensor(a)
        a = torch.tensor([])
        x0, v0 = step(x0, v0, a, pysim)

    v0 = v0 * 0
    # render_record(sim)

    for select_kp_idx in select_kp_idxs:
        config['scene']['customAttachmentVertexIdx'] = [
            (0.0, [kp_idx[select_kp_idx[0]], kp_idx[select_kp_idx[2]]])]
        sim, _, _ = set_sim_from_config(config)
        helper = diffcloth.makeOptimizeHelperWithSim("wear_hat", sim)
        pysim = pySim(sim, helper, True)

        p0 = get_coord_by_idx(x0, kp_idx[select_kp_idx[0]])
        p1 = get_coord_by_idx(x0, kp_idx[select_kp_idx[1]])
        p2 = get_coord_by_idx(x0, kp_idx[select_kp_idx[2]])
        p3 = get_coord_by_idx(x0, kp_idx[select_kp_idx[3]])

        num_points = 5
        line_points = params["line_points"]
        bend_factor = params["bend_factor"]
        point_spacing = params["point_spacing"]

        # curve_points1 = create_bent_curve(p0.detach().numpy(
        # ), p1.detach().numpy(), bend_factor=bend_factor, num_points=num_points)
        # curve_points2 = create_bent_curve(p2.detach().numpy(
        # ), p3.detach().numpy(), bend_factor=bend_factor, num_points=num_points)

        curve_points1 = create_bent_curve_spacing(p0.detach().numpy(
        ), p1.detach().numpy(), bend_factor=bend_factor, point_spacing=point_spacing)
        curve_points2 = create_bent_curve_spacing(p2.detach().numpy(
        ), p3.detach().numpy(), bend_factor=bend_factor, point_spacing=point_spacing)

        if curve_points1.shape[0] != curve_points2.shape[0]:
            max_points_num = max(
                curve_points1.shape[0], curve_points2.shape[0])
            curve_points1 = create_bent_curve(p0.detach().numpy(), p1.detach(
            ).numpy(), bend_factor=bend_factor, num_points=max_points_num)
            curve_points2 = create_bent_curve(p2.detach().numpy(), p3.detach(
            ).numpy(), bend_factor=bend_factor, num_points=max_points_num)
            num_points = max_points_num
        else:
            num_points = curve_points1.shape[0]

        for i in tqdm.tqdm(range(num_points)):

            p0_now = get_coord_by_idx(
                x0, kp_idx[select_kp_idx[0]]).detach().numpy()
            p2_now = get_coord_by_idx(
                x0, kp_idx[select_kp_idx[2]]).detach().numpy()
            p0_interpolation = np.linspace(
                p0_now, curve_points1[i], line_points)
            p2_interpolation = np.linspace(
                p2_now, curve_points2[i], line_points)

            for j in range(line_points):
                a = torch.tensor(np.concatenate(
                    (p0_interpolation[j], p2_interpolation[j])))
                x0, v0 = step(x0, v0, a, pysim)
                v0[kp_idx[select_kp_idx[0]] *
                    3:(kp_idx[select_kp_idx[0]]+1)*3] = 0
                v0[kp_idx[select_kp_idx[2]] *
                    3:(kp_idx[select_kp_idx[2]]+1)*3] = 0

            v0 = v0 * 0

        render_record(sim, [kp_idx[select_kp_idx[0]], kp_idx[select_kp_idx[1]],
                            kp_idx[select_kp_idx[2]], kp_idx[select_kp_idx[3]]], curves=[curve_points1, curve_points2])


def post_process(data_path, sample_ratio=1.5):
    npz = np.load(data_path, allow_pickle=True)
    data = npz["data"]
    kp_idx = npz["kp_idx"]
    frictional_coeff = npz["frictional_coeff"]
    k_stiff_stretching = npz["k_stiff_stretching"]
    k_stiff_bending = npz["k_stiff_bending"]

    init_state = []
    init_state_normal = []
    attached_point = []
    attached_point_target = []
    target_state = []
    target_state_normal = []
    response_matrix = []

    for d in data:
        init_state.append(d["init_state"])
        init_state_normal.append(d["init_state_normal"])
        attached_point.append(d["attached_point"])
        attached_point_target.append(d["attached_point_target"])
        target_state.append(d["target_state"])
        target_state_normal.append(d["target_state_normal"])
        response_matrix.append(d["response_matrix"])

    init_state = np.stack(init_state, axis=0)
    init_state_normal = np.stack(init_state_normal, axis=0)
    attached_point = np.stack(attached_point, axis=0)
    attached_point_target = np.stack(attached_point_target, axis=0)
    target_state = np.stack(target_state, axis=0)
    target_state_normal = np.stack(target_state_normal, axis=0)
    response_matrix = np.stack(response_matrix, axis=0)

    sample_times = int(init_state.shape[1] * sample_ratio // 2048)

    sample_data = []

    for _ in range(sample_times):
        sample_idx = np.random.randint(0, init_state.shape[1], 2048)

        sample_data_i = {}
        sample_data_i["init_state"] = init_state[:, sample_idx, :]
        sample_data_i["init_state_normal"] = init_state_normal[:, sample_idx, :]
        sample_data_i["attached_point"] = attached_point
        sample_data_i["attached_point_target"] = attached_point_target
        sample_data_i["target_state"] = target_state
        sample_data_i["target_state_normal"] = target_state_normal
        sample_data_i["response_matrix"] = response_matrix[:, :, sample_idx, :, :]

        sample_data_i["frictional_coeff"] = frictional_coeff
        sample_data_i["k_stiff_stretching"] = k_stiff_stretching
        sample_data_i["k_stiff_bending"] = k_stiff_bending
        sample_data_i["kp_idx"] = kp_idx

        sample_data.append(sample_data_i)
        pass

    return sample_data


def save_z(data, kp_idx, frictional_coeff, k_stiff_stretching, k_stiff_bending, path):
    np.savez_compressed(path, data=data, kp_idx=kp_idx, frictional_coeff=frictional_coeff,
                        k_stiff_stretching=k_stiff_stretching, k_stiff_bending=k_stiff_bending)


if __name__ == '__main__':
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

    # data, kp_idx, frictional_coeff, k_stiff_stretching, k_stiff_bending = task(
    #     params)

    post_process("DLG_Dress032_1.npz", "DLG_Dress032_1_processed.npz")
