import pickle
import numpy as np
import hdbscan


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
clusters = []
record_masks=[]
for i in range(1,20):
    cluster = unique_labels[i]
    mask_cluster = np.full(points.shape[0], False)
    mask_cluster[mask_diff] = clusters_label == cluster

    points_cluster = points[mask_cluster]
    max = np.max(points_cluster, axis=0)
    min = np.min(points_cluster, axis=0)
    bounding_max = max + 2 * (max - min)
    bounding_min = min + 2 * (min - max)
    mask_bounding = np.all((points >= bounding_min) & (points <= bounding_max), axis=1)
    record_masks.append(mask_bounding)
    # all the points in the bounding
    points_bounding = points[mask_bounding]
    colors_bounding = colors[mask_bounding]
    print(points_bounding.shape)
    #generate a numpy array with shape (n,9)
    # [x,y,z,r,g,b,label_0,label_1,mask]

    # coordinates: n x 3
    # color: n x 3
    # label_0: n x 1
    # label_1: n x 1
    # mask: n x 1 , 1 for current part,0 for others
    label1 = labels_1[0]
    label2 = labels_2[0]
    label1 = label1[mask_bounding]
    label2 = label2[mask_bounding]
    label1 = label1.reshape(-1, 1)
    label2 = label2.reshape(-1, 1)
    mask_color = mask_cluster[mask_bounding]
    mask_color = mask_color.reshape(-1, 1)
    data_bounding = np.hstack((points_bounding, colors_bounding, label1, label2, mask_color))
    np.save(f".\out\cluster_{cluster}.npy", data_bounding)
    clusters.append(data_bounding)

with open('out\\test.pkl', 'wb') as file:
    pickle.dump(clusters, file, protocol=pickle.HIGHEST_PROTOCOL)

with open('out\\test_mask.pkl', 'wb') as file:
    pickle.dump(record_masks, file, protocol=pickle.HIGHEST_PROTOCOL)