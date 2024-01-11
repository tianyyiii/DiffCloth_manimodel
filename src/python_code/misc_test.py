import time
import gdist
import trimesh
import numpy as np
import igl


def test1():
    npz = np.load("x.npz")
    vertices = npz["vert"].reshape(-1, 3).astype(np.float64)
    triangles = npz["tri"]

    points = np.random.randint(0, len(vertices), 100)

    dists = []
    argmins = []
    spent_time = []
    for i in range(10):
        max_dist = (i+1) / 10
        start = time.time()
        distances = gdist.local_gdist_matrix(
            vertices, triangles, max_distance=max_dist)
        dists.append(distances)
        distances[distances == 0] = np.inf
        mask = np.isin(range(distances.shape[1]), points)
        distances[:, ~mask] = np.inf
        nearest_point = np.argmin(distances, axis=1)
        argmins.append(np.array(nearest_point).squeeze())
        end = time.time()
        spent_time.append(end - start)
    print(dists)


def test2():
    mesh = trimesh.load_mesh("src/assets/meshes/objs/DLG_Dress032_1.obj")
    vertices = mesh.vertices
    triangles = mesh.faces.astype(np.int32)

    points = np.random.randint(0, len(vertices), 100)
    points = points.astype(np.int32)

    distances = gdist.local_gdist_matrix(
        vertices, triangles, max_distance=10)

    distances[distances == 0] = np.inf
    mask = np.isin(range(distances.shape[1]), points)
    distances[:, ~mask] = np.inf
    np1 = np.argmin(distances, axis=1)

    # src all not int points index
    # src = np.arange(len(vertices)).astype(np.int32)
    # src = np.delete(src, points)
    # result = gdist.compute_gdist(vertices, triangles, src, points, max_distance=10)

    # print(result)
    # diff = vertices[:, np.newaxis, :] - vertices[np.newaxis, :, :]
    # distances = np.sqrt(np.sum(diff**2, axis=-1))
    # distances[distances == 0] = np.inf
    # mask = np.isin(range(distances.shape[1]), points)
    # distances[:, ~mask] = np.inf
    # np2 = np.argmin(distances, axis=1)

    sampled_points = points
    # 计算所有点到每个抽样点的测地距离
    geodesic_distances = np.full((len(vertices), len(sampled_points)), np.inf)

    for i, point in enumerate(sampled_points):
        # 计算到当前抽样点的测地距离
        _, dist = igl.exact_geodesic(
            vertices, triangles, source_vertices=[point])
        geodesic_distances[:, i] = dist

    pass


if __name__ == '__main__':
    test1()
    pass
