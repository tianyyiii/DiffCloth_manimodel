from datagen_framework import show, task


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

param_template_DLNS = {
    'name': "objs/DLG_Dress032_1.obj",
    'mesh_file': "src/assets/meshes/objs/DLG_Dress032_1.obj",
    'kp_file': "src/assets/meshes/objs/kp_DLG_Dress032_1.pcd",
    'drop_step': 100,
    'select_kp_idx': [[5, 7, 0, 8]],
    "line_points": 20,
    "bend_factor": 1.5,
    "point_spacing": 0.2,
}

param_template_DLLS = {
    'name': "objs/DLLS_Dress008_0.obj",
    'mesh_file': "src/assets/meshes/objs/DLLS_Dress008_0.obj",
    'kp_file': "src/assets/meshes/objs/kp_DLLS_Dress008_0.pcd",
    'drop_step': 100,
    'select_kp_idx': [[2, 8, 0, 7], [4, 7, 1, 8]],
    "line_points": 20,
    "bend_factor": 1.5,
    "point_spacing": 0.2,
}

param_template_LT = {
    'name': "objs/DLLS_Dress008_0.obj",
    'mesh_file': "src/assets/meshes/objs/DLLS_Dress008_0.obj",
    'kp_file': "src/assets/meshes/objs/kp_DLLS_Dress008_0.pcd",
    'drop_step': 100,
    # 'select_kp_idx': [[4, 7, 1, 8]],
    'select_kp_idx': [[4, 1, 7, 8]],
    "line_points": 20,
    "bend_factor": 1.5,
    "point_spacing": 0.2,
}

DLG_name = ['DLG_Dress032_1', 'DLG_Dress033', 'DLG_Dress034', 'DLG_Dress035_0', 'DLG_Dress053', 'DLG_Dress054_0', 'DLG_Dress055_0', 'DLG_Dress079', 'DLG_dress088', 'DLG_Dress092', 'DLG_Dress095', 'DLG_Dress097', 'DLG_Dress098', 'DLG_Dress099', 'DLG_Dress100', 'DLG_Dress101', 'DLG_Dress105', 'DLG_Dress108', 'DLG_Dress118', 'DLG_Dress127', 'DLG_Dress133', 'DLG_Dress165', 'DLG_Dress196',
            'DLG_Dress197', 'DLG_Dress198', 'DLG_Dress212', 'DLG_Dress269', 'DLG_Dress309', 'DLG_Dress312', 'DLG_Dress315', 'DLG_Dress319', 'DLG_Dress365', 'DLG_Dress368', 'DLG_Dress369', 'DLG_Dress376', 'DLG_Dress377', 'DLG_Dress392', 'DLG_Dress393', 'DLG_Dress394', 'DLG_Dress400', 'DLG_Dress401', 'DLG_Dress475', 'DLG_Dress476', 'DLG_Dress477', 'DLNS_Dress020_1', 'DSNS_Dress123']

DLNS_name = ['DLNS_Dress001', 'DLNS_Dress003_0', 'DLNS_Dress003_1', 'DLNS_Dress004', 'DLNS_Dress005', 'DLNS_Dress009', 'DLNS_Dress010', 'DLNS_Dress011', 'DLNS_Dress012', 'DLNS_Dress016', 'DLNS_Dress017', 'DLNS_Dress018_0', 'DLNS_Dress019_0', 'DLNS_Dress019_1', 'DLNS_Dress020', 'DLNS_Dress021', 'DLNS_Dress022', 'DLNS_Dress027', 'DLNS_Dress027_1', 'DLNS_Dress028_0', 'DLNS_Dress030', 'DLNS_Dress031', 'DLNS_Dress038_0', 'DLNS_dress040', 'DLNS_Dress041_0', 'DLNS_Dress042_1', 'DLNS_Dress043_0', 'DLNS_Dress044_0', 'DLNS_Dress045_0', 'DLNS_Dress046_0', 'DLNS_Dress059_0', 'DLNS_Dress060', 'DLNS_Dress076', 'DLNS_Dress093', 'DLNS_Dress103', 'DLNS_Dress114', 'DLNS_Dress116', 'DLNS_Dress119', 'DLNS_Dress131',
             'DLNS_dress14', 'DLNS_Dress153', 'DLNS_Dress168', 'DLNS_Dress187', 'DLNS_Dress190', 'DLNS_Dress199', 'DLNS_Dress205', 'DLNS_Dress226', 'DLNS_Dress227', 'DLNS_Dress229', 'DLNS_dress231', 'DLNS_dress232', 'DLNS_dress236', 'DLNS_Dress240', 'DLNS_Dress241', 'DLNS_dress249', 'DLNS_Dress268', 'DLNS_Dress278', 'DLNS_Dress279', 'DLNS_Dress282', 'DLNS_Dress283', 'DLNS_Dress286', 'DLNS_Dress287', 'DLNS_Dress291', 'DLNS_Dress343', 'DLNS_Dress344', 'DLNS_Dress345', 'DLNS_Dress346', 'DLNS_Dress375', 'DLNS_Dress379', 'DLNS_dress39', 'DLNS_dress4', 'DLNS_Dress413', 'DLNS_Dress421', 'DLNS_Dress422', 'DLNS_Dress472', 'DSNS_Dress049_1', 'DSNS_Dress152', 'DSNS_Dress225', 'DSNS_Dress442', 'TNNC_Top421']

LT_name = ['PL_001', 'PL_002']

TS_name = ['TNSC_Tshirt_Ts1_0']

def gen_param_by_name(name, subpath, param_template):
    param = param_template.copy()
    param['name'] = "objs/" + subpath + "/" + name + ".obj"
    param['mesh_file'] = "src/assets/meshes/objs/" + \
        subpath + "/" + name + ".obj"
    param['kp_file'] = "src/assets/meshes/objs/" + \
        subpath + "/kp_" + name + ".pcd"
    param['x0_file'] = "src/assets/meshes/objs/" + subpath + "/" + name + "_x0.npz"
    return param


if __name__ == '__main__':
    for name in DLNS_name:
        param = gen_param_by_name(name, "DLNS", param_template_DLNS)
        show(param)
    # param = gen_param_by_name(
    #     "DLLS_Dress281", "Long_LongSleeve", param_template_DLLS)
    # param = gen_param_by_name(
    #     "DSLS_Dress025", "Short_LongSleeve", param_template_DLLS)
    # param = gen_param_by_name("DLNS_Dress001", "Long_NoSleeve", param_template_DLNS)
    # param["drop_step"] = 1
    # param = gen_param_by_name("DLLS_dress6", "Long_LongSleeve", param_template_DLNS)
    # task(param)
