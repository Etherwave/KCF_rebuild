import copy

import cv2
import numpy as np
from numpy.fft import fft2, ifft2, fftshift
from numpy import conj, real


class HOG():
    def __init__(self, winSize):
        self.winSize = winSize
        self.blockSize = (8, 8)
        self.blockStride = (4, 4)
        self.cellSize = (4, 4)
        self.nbins = 9
        self.hog = cv2.HOGDescriptor(winSize, self.blockSize, self.blockStride,
                                     self.cellSize, self.nbins)

    def get_feature(self, image):
        # 获取一张图像的hog特征
        image_h, image_w, c = image.shape
        winStride = self.winSize
        padding = (0, 0)
        hist = self.hog.compute(image, winStride, padding)
        win_w, win_h = self.winSize
        block_w, block_h = self.blockSize
        block_sw, block_sh = self.blockStride
        ans_w = (image_w//win_w)*((win_w+2*padding[0]-block_w)//block_sw+1)
        ans_h = (image_h//win_h)*((win_h+2*padding[1]-block_h)//block_sh+1)
        each_cell_ans_num = self.nbins*(self.blockSize[0]//self.cellSize[0])*(self.blockSize[1]//self.cellSize[1])
        return hist.reshape(ans_w, ans_h, each_cell_ans_num).transpose(2, 1, 0)

    def show_hog(self, hog_feature):
        c, h, w = hog_feature.shape
        feature = hog_feature.reshape(2, 2, 9, h, w).sum(axis=(0, 1))
        grid = 16
        hgrid = grid // 2
        img = np.zeros((h * grid, w * grid))
        for i in range(h):
            for j in range(w):
                for k in range(9):
                    x = int(10 * feature[k, i, j] * np.cos(np.pi / 9 * k))
                    y = int(10 * feature[k, i, j] * np.sin(np.pi / 9 * k))
                    cv2.rectangle(img, (j * grid, i * grid), ((j + 1) * grid, (i + 1) * grid), (255, 255, 255))
                    x1 = j * grid + hgrid - x
                    y1 = i * grid + hgrid - y
                    x2 = j * grid + hgrid + x
                    y2 = i * grid + hgrid + y
                    cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 1)
        cv2.imshow("img", img)
        cv2.waitKey(0)


class Tracker():
    def __init__(self):
        self.max_patch_size = 256
        self.padding = 2.5
        self.sigma = 0.6
        self.lambdar = 0.0001
        self.update_rate = 0.012
        self.gray_feature = False
        self.debug = False

    def get_feature(self, image, roi):
        # 获取图像roi那个位置的特征，并加上一个余弦窗
        cx, cy, w, h = roi
        w = int(w * self.padding) // 2 * 2
        h = int(h * self.padding) // 2 * 2
        x = int(cx - w // 2)
        y = int(cy - h // 2)

        sub_image = image[y:y + h, x:x + w, :]
        resized_image = cv2.resize(sub_image, (self.pw, self.ph))

        if self.gray_feature:
            feature = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
            feature = feature.reshape(1, self.ph, self.pw) / 255.0 - 0.5
        else:
            feature = self.hog.get_feature(resized_image)
            if self.debug:
                self.hog.show_hog(feature)

        fc, fh, fw = feature.shape
        self.scale_h = float(fh) / h
        self.scale_w = float(fw) / w

        hann2t, hann1t = np.ogrid[0:fh, 0:fw]
        hann1t = 0.5 * (1 - np.cos(2 * np.pi * hann1t / (fw - 1)))
        hann2t = 0.5 * (1 - np.cos(2 * np.pi * hann2t / (fh - 1)))
        hann2d = hann2t * hann1t

        feature = feature * hann2d
        return feature

    def gaussian_peak(self, w, h):
        # w，h是指特征图像的w，h
        # 整个函数，生成了一个目标大小的一个中心为高值，周围为低值的，一个类似余弦窗的东西
        output_sigma = 0.125
        sigma = np.sqrt(w * h) / self.padding * output_sigma
        syh, sxh = h // 2, w // 2
        y, x = np.mgrid[-syh:-syh + h, -sxh:-sxh + w]
        x = x + (1 - w % 2) / 2.
        y = y + (1 - h % 2) / 2.
        g = 1. / (2. * np.pi * sigma ** 2) * np.exp(-((x ** 2 + y ** 2) / (2. * sigma ** 2)))
        #print(np.shape(g))(50, 64)
        # g_copy = copy.deepcopy(g)
        # g_copy_min = np.min(g_copy)
        # g_copy -= g_copy_min
        # g_copy_max = np.max(g_copy)
        # g_copy /= g_copy_max
        # g_copy *= 255
        # g_copy = np.array(g_copy).astype(np.uint8)
        # cv2.imshow("1", g_copy)
        # cv2.waitKey()
        return g

    def train(self, x, y, sigma, lambdar):
        k = self.kernel_correlation(x, x, sigma)
        return fft2(y) / (fft2(k) + lambdar)

    def detect(self, alphaf, x, z, sigma):
        # 搜索图像的特征和模板图像的特征做核相关
        k = self.kernel_correlation(x, z, sigma)
        return real(ifft2(alphaf * fft2(k)))

    def kernel_correlation(self, x1, x2, sigma):
        # x1是一个图像的hog特征，x2也是一个图像的hog特征
        # 该函数要算出这两个特征图的高斯核
        # 先将x1和x2转到傅里叶空间，然后对应位置相乘
        # print(np.shape(x1))(36, 47, 64)
        # print(np.shape(x2))(36, 47, 64)
        c = ifft2(np.sum(conj(fft2(x1)) * fft2(x2), axis=0))
        # print(np.shape(c))(47, 64)
        c = fftshift(c)
        d = np.sum(x1 ** 2) + np.sum(x2 ** 2) - 2.0 * c
        k = np.exp(-1 / sigma ** 2 * np.abs(d) / d.size)
        return k

    def init(self, image, roi):
        x1, y1, w, h = roi
        cx = x1 + w // 2
        cy = y1 + h // 2
        roi = (cx, cy, w, h)

        scale = self.max_patch_size / float(max(w, h))
        self.ph = int(h * scale) // 4 * 4 + 4
        self.pw = int(w * scale) // 4 * 4 + 4
        self.hog = HOG((self.pw, self.ph))
        # 获取目标的HOG特征
        x = self.get_feature(image, roi)
        # print(np.shape(x))(36, 42, 64)
        # 获取一个类似于余弦窗的东西
        y = self.gaussian_peak(x.shape[2], x.shape[1])
        #print(np.shape(y))(42, 64)
        self.alphaf = self.train(x, y, self.sigma, self.lambdar)
        self.x = x
        self.roi = roi

    def update(self, image):
        # 先获取上一帧的位置
        cx, cy, w, h = self.roi
        max_response = -1
        # 循环5种放缩比例，用于目标的大小变换
        for scale in [0.85, 0.95, 1.0, 1.05, 1.15]:
            roi = map(int, (cx, cy, w * scale, h * scale))
            # 搜索位置的获取特征
            z = self.get_feature(image, roi)
            # 检查目标，得到相应图
            responses = self.detect(self.alphaf, self.x, z, self.sigma)
            height, width = responses.shape
            if self.debug:
                cv2.imshow("res", responses)
                cv2.waitKey(0)
            # 找到最大相应的位置
            idx = np.argmax(responses)
            res = np.max(responses)
            # 记录相应最大的结果
            if res > max_response:
                max_response = res
                dx = int((idx % width - width / 2) / self.scale_w)
                dy = int((idx / width - height / 2) / self.scale_h)
                best_w = int(w * scale)
                best_h = int(h * scale)
                best_z = z
        self.roi = (cx + dx, cy + dy, best_w, best_h)
        # 更新目标的特征模板
        self.x = self.x * (1 - self.update_rate) + best_z * self.update_rate
        y = self.gaussian_peak(best_z.shape[2], best_z.shape[1])
        # 训练新的alphaf
        new_alphaf = self.train(best_z, y, self.sigma, self.lambdar)
        # self.alphaf shape (50, 64)
        self.alphaf = self.alphaf * (1 - self.update_rate) + new_alphaf * self.update_rate

        cx, cy, w, h = self.roi
        return (cx - w // 2, cy - h // 2, w, h)

if __name__ == '__main__':
    image_path = "./1.jpg"
    image = cv2.imread(image_path)
    winSize = (100, 100)
    hog = HOG(winSize)
    feature = hog.get_feature(image)
    print(np.shape(feature))
    hog.show_hog(feature)
