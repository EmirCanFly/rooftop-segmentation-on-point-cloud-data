import numpy as np
import open3d as o3d 
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree

pcd = o3d.io.read_point_cloud("a single rooftop.pts")
o3d.visualization.draw_geometries([pcd])

#this part of the code is implemented for determining the optimal distance threshold in RANSAC algorithm
tree = KDTree(np.array(pcd.points), leaf_size=2)
nearest_dist, nearest_ind = tree.query(pcd.points, k=8)
mean_distance = np.mean(nearest_dist[:,1:])
optimal_distance_threshold = mean_distance

#this part of the code is the implementation of RANSAC algorithm and density-based clustering 
segment_models={}
segments={}
rest=pcd
number_of_max_plane = 3
for i in range(number_of_max_plane):
    colors = plt.get_cmap("tab20")(i)
    segment_models[i], inliers = rest.segment_plane(distance_threshold=optimal_distance_threshold,ransac_n=3, num_iterations=10000)
    segments[i]=rest.select_by_index(inliers)
    labels = np.array(segments[i].cluster_dbscan(eps=optimal_distance_threshold*10, min_points=10))
    candidates=[len(np.where(labels==j)[0]) for j in np.unique(labels)]
    possible_candidates = (np.unique(labels)[np.where(candidates==np.max(candidates))[0]])[0]
    best_candidate = int(possible_candidates)
    rest = rest.select_by_index(inliers, invert=True)+segments[i].select_by_index(list(np.where(labels!=best_candidate)[0]))
    segments[i]=segments[i].select_by_index(list(np.where(labels==best_candidate)[0]))
    segments[i].paint_uniform_color(list(colors[:3]))
    

o3d.visualization.draw_geometries([segments[i] for i in range(number_of_max_plane)])

#removes points that are further away from their neighbors compared to the average for the point cloud
for i in range(number_of_max_plane):
    cl, ind = segments[i].remove_statistical_outlier(nb_neighbors=20, std_ratio=2.5)
    segments[i] = cl

o3d.visualization.draw_geometries([segments[i] for i in range(number_of_max_plane)])


meshes = {}
densities = {}
for i in range(number_of_max_plane):
    segments[i].estimate_normals()
    segments[i].orient_normals_consistent_tangent_plane(100)
    mesh, density = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(segments[i], depth = 12, linear_fit = True)
    meshes[i] = mesh
    densities[i] = density 
    
for i in range(number_of_max_plane):
    vertices_to_remove = [None] * len(densities[i])
    key_value = max(densities[i]) - 0.8
    
    for j in range(len(densities[i])):
        if densities[i][j] < key_value: vertices_to_remove[j] = True
        else: False
        
    meshes[i].remove_vertices_by_mask(vertices_to_remove)

o3d.visualization.draw_geometries([meshes[i] for i in range(number_of_max_plane)])



convex_hulls = {}
for i in range(number_of_max_plane):
    colors = plt.get_cmap("tab20")(i)
    convex_hull = meshes[i].compute_convex_hull()[0]
    convex_hull.paint_uniform_color(list(colors[:3]))
    convex_hulls[i] = convex_hull
    
o3d.visualization.draw_geometries([convex_hulls[i] for i in range(number_of_max_plane)])

    