
import json
import sys
import os
import cProfile
from datagen_framework import save_z, task

# /bin/python3 ./src/python_code/start.py DLLS_dress6 Long_LongSleeve ./objs


def main():
    cloth_name = sys.argv[1]
    class_name = sys.argv[2]
    out_dir = sys.argv[3]

    # ensure the out_dir exists

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    # ensure out_dir ends with "/"
    if out_dir[-1] != "/":
        out_dir += "/"

    param_path = "src/assets/meshes/objs/" + class_name + "/" + "template.json"
    param = json.load(open(param_path, "r"))

    param['name'] = "objs/" + class_name + "/" +  cloth_name + ".obj"
    param['mesh_file'] = "src/assets/meshes/objs/" + class_name + "/" + cloth_name + ".obj"
    param['kp_file'] = "src/assets/meshes/objs/" + class_name + "/kp_" + cloth_name + ".pcd"
    param['x0_file'] = "src/assets/meshes/objs/" + class_name + "/" + cloth_name + "_x0.npz"
    
    data, kp_idx, frictional_coeff, k_stiff_stretching, k_stiff_bending = task(param)
    save_z(data, kp_idx, frictional_coeff, k_stiff_stretching, k_stiff_bending, out_dir + cloth_name)

if __name__ == "__main__":
    # cProfile.run("main()", "start.prof")
    main()
    pass