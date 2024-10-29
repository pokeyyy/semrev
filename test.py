import open3d
import numpy as np
import hdbscan
import matplotlib.pyplot as plt

data = np.load(".\datas\H2_xyzrgb_Block_1_rev.npy")

labels_1 = np.load(".\datas\H2_xyzrgb_Block_1_labels_rev.npy")
labels_2 = np.load(".\datas\H2_xyzrgb_Block_1_rev_s1_pre.npy")
mask_diff = labels_1[0] != labels_2[0]

points = data[:,0:3]
colors = data[:,3:6]
# all the points that are different
points_diff = points[mask_diff]
print(points.shape)
print(points_diff.shape)

clusters_label = hdbscan.HDBSCAN(min_cluster_size=20, min_samples=5).fit_predict(points_diff)
print(clusters_label.shape)

unique_labels = np.unique(clusters_label)
# colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))[:, :3]  # 使用matplotlib的colormap
# color_map = np.zeros((points_diff.shape[0], 3))  # 初始化颜色矩阵
#
# for i, label in enumerate(unique_labels):
#     if label == -1:
#         color_map[clusters_label == label] = [0, 0, 0]  # 噪声点为黑色
#     else:
#         color_map[clusters_label == label] = colors[i]  # 其他簇使用不同颜色

print("cluster的数量：{}".format(unique_labels.size))

# single cluster
cluster = unique_labels[1]
mask_cluster = clusters_label == cluster
points_cluster = points_diff[mask_cluster]
max = np.max(points_cluster, axis=0)
min = np.min(points_cluster, axis=0)
bounding_max = max+2*(max-min)
bounding_min = min+2*(min-max)
mask_bounding = np.all((points >= bounding_min) & (points <= bounding_max), axis=1)
# all the points in the bounding
points_bounding = points[mask_bounding]
colors_bounding = colors[mask_bounding]
print(points_bounding.shape)

vis = open3d.visualization.Visualizer()
vis.create_window()

# grey and purple
mask_color = (points_bounding[:, None] == points_cluster).all(-1).any(-1)
color_show = np.empty(colors_bounding.shape)
color_show[:] = [0.50196, 0.50196, 0.50196]
color_show[mask_color] = [0.5, 0.0, 0.5]

pcd = open3d.geometry.PointCloud()
pcd.points = open3d.utility.Vector3dVector(points_bounding)
pcd.colors = open3d.utility.Vector3dVector(color_show)

# 输出测试
label1 = labels_1[0]
label2 = labels_2[0]
label1 = label1[mask_bounding]
label2 = label2[mask_bounding]
print(label1.shape)
label1 = label1.reshape(-1,1)
label2 = label2.reshape(-1,1)
print(label1.shape)
mask_color = mask_color.reshape(-1,1)
data_bounding = np.hstack((points_bounding, colors_bounding, label1, label2,mask_color))
np.save(".\out\cluster_6.npy", data_bounding)
print(data_bounding.shape)

vis.add_geometry(pcd)

vis.run()

# re = input('请输入修改后的标签:')
# mask_labels_rev = (points[:, None] == points_cluster).all(-1).any(-1)
# print(mask_labels_rev.shape)
# labels_out = labels_1
# labels_out[1][mask_labels_rev] = re

vis.destroy_window()

