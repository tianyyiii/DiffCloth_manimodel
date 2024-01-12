from datagen_framework import save_z, task
from manual_select_params import gen_param_by_name

param_template_DLG = {
    'name': "objs/DLG_Dress032_1.obj",
    'mesh_file': "src/assets/meshes/objs/DLG_Dress032_1.obj",
    'kp_file': "src/assets/meshes/objs/kp_DLG_Dress032_1.pcd",
    'drop_step': 100,
    'select_kp_idx': [[5, 7, 0, 8]],
    "line_points": 20,
    "bend_factor": 1.5,
    "point_spacing": 0.2,
}

DLG_name = ['DLG_Dress032_1', 'DLG_Dress033', 'DLG_Dress034', 'DLG_Dress035_0', 'DLG_Dress053', 'DLG_Dress054_0', 'DLG_Dress055_0', 'DLG_Dress079', 'DLG_dress088', 'DLG_Dress092', 'DLG_Dress095', 'DLG_Dress097', 'DLG_Dress098', 'DLG_Dress099', 'DLG_Dress100', 'DLG_Dress101', 'DLG_Dress105', 'DLG_Dress108', 'DLG_Dress118', 'DLG_Dress127', 'DLG_Dress133', 'DLG_Dress165', 'DLG_Dress196', 'DLG_Dress197', 'DLG_Dress198', 'DLG_Dress212', 'DLG_Dress269', 'DLG_Dress309', 'DLG_Dress312', 'DLG_Dress315', 'DLG_Dress319', 'DLG_Dress365', 'DLG_Dress368', 'DLG_Dress369', 'DLG_Dress376', 'DLG_Dress377', 'DLG_Dress392', 'DLG_Dress393', 'DLG_Dress394', 'DLG_Dress400', 'DLG_Dress401', 'DLG_Dress475', 'DLG_Dress476', 'DLG_Dress477', 'DLNS_Dress020_1', 'DSNS_Dress123']

def process_task(name):
    param = gen_param_by_name(name, "DLNS", param_template_DLG)
    data, kp_idx, frictional_coeff, k_stiff_stretching, k_stiff_bending = task(param)
    save_z(data, kp_idx, frictional_coeff, k_stiff_stretching, k_stiff_bending, name)

if __name__ == '__main__':
    out_path = "./"
    name = DLG_name[0]
    
    param = gen_param_by_name(name, "DLG", param_template_DLG)
    data, kp_idx, frictional_coeff, k_stiff_stretching, k_stiff_bending = task(param)
    save_z(data, kp_idx, frictional_coeff, k_stiff_stretching, k_stiff_bending, out_path + name)
    
    pass