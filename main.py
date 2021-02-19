# Java: https://github.com/Andrew-Tuen/umeyama_java/blob/master/umeyama_jama.java
# Transform Matrix: https://upload.wikimedia.org/wikipedia/commons/2/2c/2D_affine_transformation_matrix.svg
import cv2
import numpy as np
from skimage import transform as trans

# Jason's landmark in pic
lmk = np.array(
    [[739, 121], [812, 114], [774, 159], [743, 202], [815, 195]],
    dtype=np.float32)

target = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32)


def main():
    # 用 skimage call
    print("by skimage")
    tform = trans.SimilarityTransform()
    tform.estimate(lmk, target)
    matrix = tform.params[0:2, :]
    print(matrix)
    """
    [[ 4.73931668e-01 -2.83592859e-02 -3.07542722e+02]
     [ 2.83592859e-02  4.73931668e-01 -2.50990295e+01]]
    """

    # 用 code call
    print("by code")
    matrix = code(lmk, target)
    matrix = matrix[0:2, :]
    print(matrix)
    """
    [[ 4.73931668e-01 -2.83592859e-02 -3.07542722e+02]
     [ 2.83592859e-02  4.73931668e-01 -2.50990295e+01]]
    """

    # 將旋轉矩陣matrix 套用到圖片後的結果
    img = cv2.imread('jason.jpg')
    image_size = 112
    rotate_img = cv2.warpAffine(img, matrix, (image_size, image_size), borderValue=0.0)
    cv2.imwrite('jason-rotate.jpg', rotate_img)


def code(src, dst):
    num = src.shape[0]
    dim = src.shape[1]

    # Compute mean of src and dst.
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)

    # Subtract mean from src and dst.
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    # Eq. (38).
    A = np.dot(dst_demean.T, src_demean / num)

    # Eq. (39).
    d = np.ones((dim,), dtype=np.double)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1

    T = np.eye(dim + 1, dtype=np.double)

    U, S, V = np.linalg.svd(A)

    # Eq. (40) and (43).
    rank = np.linalg.matrix_rank(A)
    if rank == 0:
        return np.nan * T
    elif rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(V) > 0:
            T[:dim, :dim] = np.dot(U, V)
        else:
            s = d[dim - 1]
            d[dim - 1] = -1
            T[:dim, :dim] = np.dot(np.dot(U, np.diag(d)), V)
            d[dim - 1] = s
    else:
        T[:dim, :dim] = np.dot(np.dot(U, np.diag(d)), V)

        # Eq. (41) and (42).
    scale = 1.0 / src_demean.var(axis=0).sum() * np.dot(S, d)

    T[:dim, dim] = dst_mean - scale * (np.dot(T[:dim, :dim], src_mean.T))
    T[:dim, :dim] *= scale

    return T


if __name__ == '__main__':
    main()
