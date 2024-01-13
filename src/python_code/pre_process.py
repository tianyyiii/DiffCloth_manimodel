import os 
import trimesh
import pymeshfix
import shutil

def process_mesh(base_path, cloth_path, save_path, rotation):
    save_path = save_path + "/" + cloth_path
    cloth_path = base_path + "/" + cloth_path
    index = 0
    for cloth in os.listdir(cloth_path):
        cloth_mesh = cloth_path + "/" + cloth + "/" + cloth + ".obj"
        try:
            scene = trimesh.load_mesh(cloth_mesh)
        except:
            continue
        try: 
            mesh = scene.dump()[1]
        except:
            mesh = scene
        while(len(mesh.vertices) > 1000):
            mesh = mesh.simplify_quadric_decimation(face_count=0.8*len(mesh.faces))
        # tin = pymeshfix.PyTMesh()
        # tin.load_array(mesh.vertices, mesh.faces)
        # #tin.join_closest_components()
        # #tin.fill_small_boundaries(nbe=10)
        # #tin.clean(max_iters=10, inner_loops=3)
        # clean_points, clean_faces = tin.return_arrays()
        #clean_points, clean_faces = pymeshfix.clean_from_arrays(mesh.vertices, mesh.faces)
        #mesh = trimesh.Trimesh(vertices=clean_points, faces=clean_faces)
        connected_components = mesh.split(only_watertight=False)
        iteration = 0
        print(len(connected_components))
        while len(connected_components) != 1:
            iteration += 1 
            trimesh.repair.fill_holes(mesh)
            connected_components = mesh.split(only_watertight=False)
            if iteration == 10:
                break
        if len(connected_components) == 1:
            vertices = mesh.vertices
            if rotation:
                vertices[:, [1,2]] = vertices[:, [2, 1]]
            mesh.vertices = vertices
            min_bound, max_bound = mesh.bounding_box.bounds
            scale = max(max_bound - min_bound)
            vertices = vertices/scale * 4 
            mesh.vertices = vertices 
            min_bound, max_bound = mesh.bounding_box.bounds
            average = (min_bound + max_bound) / 2 
            for i in range(3):
                vertices[:, i] -= average[i]
            mesh.vertices = vertices
            min_bound, max_bound = mesh.bounding_box.bounds
            vertices[:, 1] -= (min_bound[1]-(-5.65))
            mesh.vertices = vertices
            os.makedirs(save_path + "/" + str(index), exist_ok=True)
            mesh.export(save_path + "/" + str(index) + "/" + "mesh.obj")
            shutil.copyfile(cloth_path + "/" + cloth + "/" + "kp_" + cloth + ".pcd", save_path + "/" + str(index) + "/" + "kp.pcd")

            index += 1
    
if __name__ == "__main__":
    base_path = "../../../ClothesNetData/ClothesNetM"
    save_path = "../../../ManiModelData/ClothesNetM"
    clothes = os.listdir(base_path + "/" + "Tops")
    clothes = ["Tops/" + cloth for cloth in clothes]
    for cloth in clothes:
        process_mesh(base_path, cloth, save_path, rotation=False)
    