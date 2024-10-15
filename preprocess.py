import open3d
import numpy as np
import pickle

if __name__ == '__main__':
    with open('out/test.pkl', 'rb') as file:
        clusters = pickle.load(file)

    with open('out/test_mask.pkl', 'rb') as file:
        masks = pickle.load(file)

    num = masks[1].shape[0]
    print(num)
    labels_out = np.zeros(num)
    for i in range(len(clusters)):
        mask = masks[i]
        cluster = clusters[i]
        labels_out[mask] = cluster[:,-2]

    labels_out = labels_out.reshape(1, -1)
    labels_out = np.vstack((labels_out, np.zeros((1,num))))
    print(labels_out.shape)
    np.save('out/test_labels_out_pre.npy', labels_out)