
import json
import os
from datagen_framework import random_task, save_z


if __name__ == '__main__':

    # cloth_name = sys.argv[1]
    # class_name = sys.argv[2]
    # out_dir = sys.argv[3]

    cloth_name = "DST_Dress013"
    class_name = "Short_Tube"
    out_dir = "./objs"

    cloth_name = cloth_name.strip()
    class_name = class_name.strip()
    out_dir = out_dir.strip()

    # ensure the out_dir exists

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # ensure out_dir ends with "/"
    if out_dir[-1] != "/":
        out_dir += "/"

    param_path = "src/assets/meshes/objs/" + class_name + "/" + "template.json"
    param = json.load(open(param_path, "r"))

    param['name'] = "objs/" + class_name + "/" + cloth_name + ".obj"
    param['mesh_file'] = "src/assets/meshes/objs/" + \
        class_name + "/" + cloth_name + ".obj"
    param['kp_file'] = "src/assets/meshes/objs/" + \
        class_name + "/kp_" + cloth_name + ".pcd"
    param['x0_file'] = "src/assets/meshes/objs/" + \
        class_name + "/" + cloth_name + "_x0.npz"

    task_out_file = "src/assets/meshes/objs/" + \
        class_name + "/" + cloth_name + ".npz"

    data, kp_idx, frictional_coeff, k_stiff_stretching, k_stiff_bending = random_task(
        param, task_out_file)

    save_z(data, kp_idx, frictional_coeff, k_stiff_stretching,
           k_stiff_bending, out_dir + cloth_name)

    pass
