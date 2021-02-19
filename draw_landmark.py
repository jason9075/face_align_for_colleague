import cv2
import numpy as np


def main():
    lmk = np.array(
        [[739, 121], [812, 114], [774, 159], [743, 202], [815, 195]],
        dtype=np.float32)

    img = cv2.imread('jason.jpg')

    for l in lmk:
        x = l[0]
        y = l[1]
        cv2.circle(img, (x, y), 5, (255, 0, 0), -1)

    cv2.imwrite('jason-with-landmark.jpg', img)


if __name__ == '__main__':
    main()
