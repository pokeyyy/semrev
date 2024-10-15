import open3d
import numpy as np
import pickle

if __name__ == '__main__':
    with open('out/test_label_pre.pkl', 'rb') as file:
        clusters = pickle.load(file)

    with open('out/test_mask.pkl', 'rb') as file:
        masks = pickle.load(file)

    with open('out/test.pkl', 'rb') as file:
        clusters_ori = pickle.load(file)

    num = masks[1].shape[0]
    print(num)
    labels_out = np.zeros(num)
    t = 0 #记录选择预测的个数
    for i in range(len(clusters)):
        mask = masks[i]
        cluster = clusters[i]
        cluster = cluster.reshape(-1)
        labels_out[mask] = cluster
        cluster_ori = clusters_ori[i]
        cluster_ori = cluster_ori[:,-3]
        if np.array_equal(cluster_ori, cluster):
            t+=1

    labels_out = labels_out.reshape(1, -1)
    labels_out = np.vstack((labels_out, np.zeros((1,num))))
    print(labels_out.shape)
    print(t)
    np.save('out/test_labels_out_rev.npy', labels_out)