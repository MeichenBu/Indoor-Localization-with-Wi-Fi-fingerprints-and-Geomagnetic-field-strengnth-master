import pandas as pd
import numpy as np
import os
import math
import matplotlib.pyplot as plt
from cv2 import dct

PI = math.pi


def DCT_2D():
    ls = []
    folder_path = 'AE_output//'
    folder = os.listdir(folder_path)
    for file in folder:
        [image_size, type] = file.split('.')
        N = int(image_size)
        print(N)
        image_path = os.path.join(folder_path, file)
        img = pd.read_csv(image_path, index_col=0)
        img = np.asarray(img.apply(lambda x: x.mean())).reshape(N, N)
        arr = img.astype(np.float32)
        arr_dct = np.zeros((N, N))
        for u in range(N):
            for v in range(N):
                sum = 0
                for x in range(N):
                    for y in range(N):
                        sum = sum + arr[x, y] * np.cos(((2.0 * x + 1) * u * PI) / (2.0 * N)) * np.cos(
                            ((2.0 * y + 1) * v * PI) / (2.0 * N))
                if (u == 0):
                    Cu = np.sqrt(1.0 / N)
                else:
                    Cu = np.sqrt(2.0 / N)
                if (v == 0):
                    Cv = np.sqrt(1.0 / N)
                else:
                    Cv = np.sqrt(2.0 / N)
                arr_dct[u, v] = Cu * Cv * sum
        # arr_dct = cv2.dct(arr)  #if the opencv dct function is used
        ls.append(arr_dct)
        plt.imshow(arr_dct, cmap='gray')
        plt.title('DCT Figure' + ' (Size = ' + str(N) + 'x' + str(N) + ')')
        plt.colorbar()
        plt.show()


if __name__ == '__main__':
    DCT_2D()
