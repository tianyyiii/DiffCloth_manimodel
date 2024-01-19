
import json
import os

from tqdm import tqdm
from datagen_framework import random_task, save_z


def main(cloth_name, class_name, out_dir="./objs/random/"):
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
           k_stiff_bending, out_dir + cloth_name + "_random.npz")


def process_dir(path, class_name):
    
    # read path/names.txt
    with open(path + "/names.txt", "r") as f:
        names = f.readlines()
        names = [name.strip() for name in names]
    
    useable_names = []
    
    for n in names:
        if os.path.exists(path + "/" + n + ".npz"):
            useable_names.append(n)
    
    for n in tqdm(useable_names, desc="RandomGen:" + class_name):
        main(n, class_name)

    pass


if __name__ == '__main__':

    # cloth_name = sys.argv[1]
    # class_name = sys.argv[2]
    # out_dir = sys.argv[3]

    process_dir("src/assets/meshes/objs/Long_LongSleeve", "Long_LongSleeve")
    process_dir("src/assets/meshes/objs/Long_NoSleeve", "Long_NoSleeve")
    process_dir("src/assets/meshes/objs/Long_ShortSleeve", "Long_ShortSleeve")
    process_dir("src/assets/meshes/objs/Long_Tube", "Long_Tube")
    process_dir("src/assets/meshes/objs/Short_Gallus", "Short_Gallus")
    process_dir("src/assets/meshes/objs/Short_NoSleeve", "Short_NoSleeve")
    process_dir("src/assets/meshes/objs/Short_ShortSleeve", "Short_ShortSleeve")
    process_dir("src/assets/meshes/objs/Short_Tube", "Short_Tube")

    pass
