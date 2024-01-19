import os
from datagen_framework import set_sim_from_config, read_mesh_ignore_vtvn, calculate_vertex_normal
import yaml
import numpy as np
from jacobian import full_jacobian
import torch

CONFIG_TEMPLATE = {
    'fabric': {
        "clothDimX": 6,
        "clothDimY": 6,
        "k_stiff_stretching": 5000,
        "k_stiff_bending":  1.5,
        "gridNumX": 40,
        "gridNumY": 80,
        "density": 1.5,
        "keepOriginalScalePoint": True,
        'isModel': True,
        "custominitPos": False,
        "fabricIdx": 2,  # Enum Value
        "color": (0.3, 0.9, 0.3),
        "name":  "remeshed/top.obj",
    },
    'scene': {
        "orientation": 0,  # Enum Value
        "attachmentPoints": 2,  # CUSTOM_ARRAY
        "customAttachmentVertexIdx": [(0., [])],
        "trajectory": 0,  # Enum Value
        "primitiveConfig": 3,  # Enum Value
        'windConfig': 0,  # Enum Value
        'camPos':  (-10.38, 4.243, 12.72),
        "camFocusPos": (0, -4, 0),
        'camFocusPointType': 3,  # Enum Value
        "sceneBbox":  {"min": (-7, -7, -7), "max": (7, 7, 7)},
        "timeStep": 0.01,
        "stepNum": 40,
        "forwardConvergenceThresh": 1e-8,
        'backwardConvergenceThresh': 5e-4,
        'name': "wind_tshirt"
    }
}


def get_config_from_yaml(yaml_path, obj_name):
    with open(yaml_path, 'r') as f:
        yml_config = yaml.load(f, Loader=yaml.FullLoader)
        config = CONFIG_TEMPLATE.copy()
        config['fabric']['name'] = obj_name
        config['scene']['name'] = yml_config['scene_config']['name']
        config["scene"]["customAttachmentVertexIdx"] = [
            (0., yml_config["scene_config"]["customAttachmentVertexIdx"])]
    return config


def get_f_kp_kd_from_yaml(yaml_path):
    with open(yaml_path, 'r') as f:
        yml_config = yaml.load(f, Loader=yaml.FullLoader)
        frictional_coeff = 0.01
        k_stiff_bending = yml_config['fabric_config']["k_stiff_bending"]
        k_stiff_stretching = yml_config['fabric_config']["k_stiff_stretching"]
        attachment_points = yml_config['scene_config']['customAttachmentVertexIdx']
    return frictional_coeff, k_stiff_bending, k_stiff_stretching, attachment_points


def process_one_traj(path):
    data = {}

    all_files = os.listdir(path)
    yml_file = [f for f in all_files if f.endswith('.yaml')][0]

    step0_obj = os.path.join(path, "step0.obj")
    cliped_path = step0_obj[step0_obj.find("remeshed"):]

    vertices, faces = read_mesh_ignore_vtvn(step0_obj)

    keypoints = np.random.choice(vertices.shape[0], 10, replace=False)
    frictional_coeff, k_stiff_bending, k_stiff_stretching, attachment_points = get_f_kp_kd_from_yaml(
        os.path.join(path, yml_file))

    data["keypoints"] = keypoints
    data["frictional_coeff"] = frictional_coeff
    data["k_stiff_bending"] = k_stiff_bending
    data["k_stiff_stretching"] = k_stiff_stretching

    data["init_state"] = vertices
    data["init_state_normal"] = calculate_vertex_normal(vertices, faces)

    data["target_state"] = []
    data["target_state_normal"] = []
    data["attached_point"] = []
    data["attached_point_target"] = []

    for i in range(40):
        step_txt = os.path.join(path, f"step{i}.txt")
        x = np.loadtxt(step_txt)
        data["target_state"].append(x)
        data["target_state_normal"].append(calculate_vertex_normal(x, faces))

        if len(attachment_points) > 7:
            data["attached_point"].append(
                [attachment_points[0], attachment_points[7]])
            data["attached_point_target"].append(
                np.vstack([x[attachment_points[0]], x[attachment_points[7]]]))
        else:
            data["attached_point"].append([attachment_points[0], None])
            data["attached_point_target"].append(np.vstack(
                [x[attachment_points[0]], np.zeros_like(x[attachment_points[0]])]))

    config = get_config_from_yaml(os.path.join(path, yml_file), cliped_path)
    x = torch.tensor(x.flatten())
    v = torch.zeros_like(x)
    jacobian = full_jacobian(
        vertices, faces, x, v, keypoints, config)
    data["response_matrix"] = jacobian

    return data


if __name__ == '__main__':
    # config = get_config_from_yaml('src/assets/meshes/remeshed/standard_tie/ep1/step0/traj0/traj_config.yaml', "remeshed/standard_tie/ep1/step0/traj0/step0.obj")

    # set_sim_from_config(config)

    data = process_one_traj("src/assets/meshes/remeshed/standard_tie/ep1/step0/traj0")

    pass
