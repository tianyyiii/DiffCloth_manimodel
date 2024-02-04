import numpy as np
import trimesh
from trimesh import sample
import gdist
import random
import os

def visualize(mesh):
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
    if mesh.visual.vertex_colors is not None:
        vertex_colors_normalized = mesh.visual.vertex_colors[:, :3] / 255.0
        o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors_normalized)
    o3d_mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([o3d_mesh])


def apply_gradient_color(mesh, indices, stage, max_distance=0.2, transparency=0.5):
    colors = np.ones((len(mesh.vertices), 4)) * np.array([1.0, 1.0, 1.0, transparency])     
    if stage == "init":
        color_w = np.array([229/255,162/255,32/255,1])
    else:
        color_w = np.array([81/255,141/255,178/255,1])
    rf = []
    for point_idx in indices:
        vertices = np.array(mesh.vertices, dtype=np.float64)
        faces = np.array(mesh.faces, dtype=np.int32)
        distances = gdist.compute_gdist(vertices, faces, source_indices=np.array([point_idx], dtype=np.int32))
        rf.append(distances)
    if len(rf) == 2:
        distances = np.min(np.vstack((rf[0], rf[1])), axis=0)
    normalized_distances = np.clip(distances / max_distance, 0, 1)
    for i, dist in enumerate(normalized_distances):
        if dist < 1:
            colors[i, :] = np.array([251/255,68/255,91/255,1])
        else:
            colors[i, :] = color_w
    mesh.visual.vertex_colors = colors
    return mesh

def create_arrow(radius=0.03, height=0.4, cylinder_height_ratio=0.7):
    cylinder_height = height * cylinder_height_ratio
    shaft = trimesh.creation.cylinder(radius=radius, height=cylinder_height, sections=32)
    head_height = height - cylinder_height
    head = trimesh.creation.cone(radius=radius * 2, height=head_height, sections=32)
    head.apply_translation([0, 0, cylinder_height / 2])
    arrow = trimesh.util.concatenate(shaft, head)
    arrow.apply_translation([0, 0, cylinder_height / 2])
    return arrow

def apply_arrow(mesh, attached_points, directions):
    arrow_meshes = []
    
    for i, attached_point in enumerate(attached_points):
        pos = mesh.vertices[attached_point]
        direction = directions[i]
        arrow= create_arrow()
        align_matrix = trimesh.geometry.align_vectors([0, 0, 1], direction)
        arrow.apply_transform(align_matrix)
        arrow.apply_translation(pos)
        colors = np.ones((len(arrow.vertices), 4)) * np.array([39/255,115/255,255/255,1]) 
        arrow.visual.vertex_colors = colors
        arrow_meshes.append(arrow)
    combined_arrows = trimesh.util.concatenate(*arrow_meshes)

    return combined_arrows


def normalize(scene1, scene2):
    min_bound, max_bound = scene1.bounds
    scale_factor = 1.0 / max(max_bound - min_bound)
    scaling_matrix = trimesh.transformations.scale_matrix(scale_factor, [0, 0, 0])
    scene1.apply_transform(scaling_matrix)
    scene2.apply_transform(scaling_matrix)
    new_min, new_max = scene1.bounds
    new_centroid = (new_max + new_min) / 2.0
    translation_matrix = trimesh.transformations.translation_matrix(-new_centroid)
    scene1.apply_transform(translation_matrix) 
    scene2.apply_transform(translation_matrix)
    return scene1, scene2


    

if __name__ == "__main__":
    base_dir = "/home/ubuntu/middle/ManiModelData/ClothesNetM/Tops"
    cloth_dir = "/home/ubuntu/ManiModelData/ClothesNetM/Tops"
    num = 30
    
    
    tops = os.listdir(base_dir)
    for top in tops:
        paths = os.listdir(base_dir + "/" + top)
        if len(paths) > num:
            paths = random.sample(paths, num)
        for path in paths:
            data_path = f"{base_dir}/{top}/{path}/cloth.npz"
            data = np.load(data_path)
            obj = path.split("_")[0]
            mesh_path = f"{cloth_dir}/{top}/{obj}/mesh.obj"
            stage_length = data["init_state"].shape[0]
            attached_points = data["attached_point"][0, 0, :]
            if stage_length > 6:
                start_stage = random.randint(0, stage_length-7)
                init_state = data["init_state"][start_stage, :, :]
                target_state1 = data["init_state"][start_stage+3, :, :]
                target_state2 = data["init_state"][start_stage+6, :, :]
                motions1 = data["attached_point_target"][start_stage, 0, :, :]
                motions2 = data["attached_point_target"][start_stage+3, 0, :, :]
                points0 = []
                d0 = np.zeros((2, 3))
                for j, attached_point in enumerate(attached_points):
                    if attached_point >= 0:
                        points0.append(int(attached_point))
                        d0[j, :] = motions1[j, :] - init_state[int(attached_point), :]
                if (d0[0, :] == 0).all():
                    d0[0, :] = d0[1, :]
                points1 = []
                d1 = np.zeros((2, 3))
                for j, attached_point in enumerate(attached_points):
                    if attached_point >= 0:
                        points1.append(int(attached_point))
                        d1[j, :] = motions2[j, :] - target_state1[int(attached_point), :]
                if (d1[0, :] == 0).all():
                    d1[0, :] = d1[1, :]
                
                mesh0 = trimesh.load_mesh(mesh_path)
                mesh0.vertices = init_state
                mesh0 = mesh0.subdivide()
                mesh0 = apply_gradient_color(mesh0, points0, "init")
                arrow0 = apply_arrow(mesh0, points0, d0)
                cloud0 = trimesh.points.PointCloud(vertices=mesh0.vertices, colors=mesh0.visual.vertex_colors)
                mesh1 = trimesh.load_mesh(mesh_path)
                mesh1.vertices = target_state1
                mesh1 = mesh1.subdivide()
                mesh1 = apply_gradient_color(mesh1, points1, "target")
                arrow1 = apply_arrow(mesh1, points1, d1)
                cloud1 = trimesh.points.PointCloud(vertices=mesh1.vertices, colors=mesh1.visual.vertex_colors)
                mesh2 = trimesh.load_mesh(mesh_path)
                mesh2.vertices = target_state2
                mesh2 = mesh2.subdivide()
                mesh2 = apply_gradient_color(mesh2, points1, "target")
                cloud2 = trimesh.points.PointCloud(vertices=mesh2.vertices, colors=mesh2.visual.vertex_colors)          
                scene1 = arrow0 + arrow1
                inverted = scene1.copy()
                inverted.invert()
                scene1 = trimesh.util.concatenate(scene1, inverted)
                scene2 = cloud0 + cloud1 + cloud2
                scene1, scene2 = normalize(scene1, scene2)
                os.makedirs(f"output/{top}/{path}", exist_ok=True)
                scene1.export(f"output/{top}/{path}/arrow.ply")
                scene2.export(f"output/{top}/{path}/pc.ply")