import numpy as np
import imutils
import sys
import cv2
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QDialog, QApplication, QMainWindow
from PyQt5.uic import loadUi
from cv2 import *
import tkinter as tk
from tkinter import filedialog
from typing import Union,List
import os
from matplotlib import pyplot as plt
from sklearn.compose import make_column_selector


class ShowImage(QMainWindow):
    def __init__(self):
        super(ShowImage, self).__init__()
        loadUi('cbir.ui', self)
        self.image = None
        self.query = None
        self.load.clicked.connect(self.loadImg)
        self.proses.clicked.connect(self.main_process)


    @pyqtSlot()

    def loadImg(self):
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(
                parent=root, initialdir='C:/Tutorial',
                title='Choose file',
                filetypes=[('Image Files',('*.jpg','*.jpeg','*.png','*.bmp'))])
        print(file_path)
        self.image = cv2.imread(file_path)
        self.query = self.preprocess(self.image)
        self.showimage()


    def showimage(self):
        qtformat = QImage.Format_Indexed8
        if len(self.image.shape) - -3:
            if (self.image.shape[2]) == 4:
                qtformat = QImage.Format_RGBA8888
            else:
                qtformat = QImage.Format_RGB888
        img = QImage(self.image, self.image.shape[1], self.image.shape[0], self.image.strides[0], qtformat)
        img = img.rgbSwapped()
        h, w = self.label.height(), self.label.width()
        self.label.setPixmap(QPixmap.fromImage(img).scaled(h,w,Qt.KeepAspectRatio))
        self.label.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)

    def preprocess(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)     #grayscaling citra
        orb = cv2.ORB_create()                              #membuat objek ORB
        kp1, des1 = orb.detectAndCompute(image, None)       #eksekusi deteksi interest point

        return des1

    def main_process(self):
        list_data = []
        x = 0
        for file in os.listdir('image'):                    # mengambil ddata dari folder
            img1 = cv2.imread('image/'+file)                # membaca file citra
            list_data.append([])
            list_data[x].append(file)                       # memasukan nama file ke array
            list_data[x].append(img1)                       # memasukan citra RGB ke array
            list_data[x].append(self.preprocess(img1))      # memasukan citra hasil ekstrasi ke array
            x += 1


        print("Data Loaded..")
        print('=================================')
        print("Matching Proccess..")

        # list_data[nama file, image, features]

        list_data = np.array(list_data,dtype=np.object)     # (numpy) mengubah bentuk array
        list_data = np.hstack((list_data, np.zeros((list_data.shape[0], 1), dtype=list_data.dtype)))    # menambahkan kolom untuk hasil

        # list_data[nama file, image, features, zeros]

        for y in range(len(list_data)):
            list_data[y,3] = self.matching(list_data[y, 2])     # eksekusi proses pencocokan dan masukan ke array

        # list_data[nama file, image, features, jumlah cocok]

        print("Matching Proccess Done")

        print('=================================')

        print("Sorting Result..")

        list_data = list_data[np.argsort(list_data[:, 3])[::-1]]        # pengurutan hasil pencocokan dari nilai yg terbesar

        print('Showing Result......')
        self.show_images2(list_data[:, 1],6,3)

    def matching(self, des):
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(self.query, des)
        matches = sorted(matches, key=lambda x: x.distance)
        return len(matches)


    def show_images(self, images: List[np.ndarray]) -> None:
        n: int = len(images)
        f = plt.figure()
        for i in range(n):
            f.add_subplot(1, n, i + 1)
            plt.imshow(images[i])

        plt.show(block=True)

    def show_images2(self, images, rows = 1, cols=1):
        figure, ax = plt.subplots(nrows=rows, ncols=cols)
        for ind in range(len(images)):
            ax.ravel()[ind].imshow(images[ind])
            ax.ravel()[ind].set_axis_off()
        plt.tight_layout()
        plt.show()


app = QtWidgets.QApplication(sys.argv)

window = ShowImage()
window.setWindowTitle('CBIR using ORB')
window.show()
sys.exit(app.exec_())