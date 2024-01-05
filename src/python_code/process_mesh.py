import os 
import trimesh
import shutil

def process_mesh(base_path, cloth_path, save_path):
    save_path = save_path + "/" + cloth_path
    cloth_path = base_path + "/" + cloth_path
    print(cloth_path)
    print(save_path)
    index = 0
    for cloth in os.listdir(cloth_path):
        cloth_mesh = cloth_path + "/" + cloth + "/" + cloth + ".obj"
        scene = trimesh.load_mesh(cloth_mesh)
        try: 
            mesh = scene.dump()[1]
        except:
            mesh = scene
        connected_components = mesh.split(only_watertight=False)
        if len(connected_components) == 1:
            vertices = mesh.vertices
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
            #if len(mesh.vertices) > 2048:
            #    mesh = mesh.simplify_quadratic_decimation(2048)
            os.makedirs(save_path + "/" + str(index), exist_ok=True)
            mesh.export(save_path + "/" + str(index) + "/" + "mesh.obj")
            shutil.copyfile(cloth_path + "/" + cloth + "/" + "kp_" + cloth + ".pcd", save_path + "/" + str(index) + "/" + "kp.pcd")
            
            index += 1
    
if __name__ == "__main__":
    base_path = "../../../ClothesNetData/ClothesNetM"
    save_path = "../../../ManiModelData/ClothesNetM"
    clothes = ["Hat", "Socks/Long", "Skirt/Long", "Trousers/Short"]
    for cloth in ["Trousers/Long", "Tops/NoCollar_Lsleeve_FrontOpen", 
                  "Tops/Collar_Lsleeve_FrontClose", "Dress/Long_LongSleeve"]:
        process_mesh(base_path, cloth, save_path)
    