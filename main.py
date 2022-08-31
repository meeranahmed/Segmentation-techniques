from PyQt5.QtWidgets import QMainWindow,QApplication,QPushButton,QLabel,QFileDialog,QTabWidget,QComboBox
from PyQt5.QtGui import QPixmap
from sympy import im
from mainwindow import Ui_MainWindow
from PyQt5 import uic
import cv2
import sys
import numpy as np
from Utilities.meanShift import *
from Utilities.threshold import *
from Utilities.region_growing import *
from Utilities.Agg import *
from Utilities.Kmeans import *
from Utilities.BGR_to_LUV import *

class UI(QMainWindow):
    def __init__(self):
        super(UI,self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.open1 = self.findChild(QPushButton,"OpenImage")
        self.open2 = self.findChild(QPushButton,"OpenImage_2")
        self.out1 = self.findChild(QPushButton,"OutputImage")
        self.out2 = self.findChild(QPushButton,"OutputImage_2")
        self.inputimg1=self.findChild(QLabel,"Input")
        self.inputimg2=self.findChild(QLabel,"Input_2")
        self.outputimg1=self.findChild(QLabel,"Output")
        self.outputimg2=self.findChild(QLabel,"Output_2")
        self.tab=self.findChild(QTabWidget,"tabWidget")
        self.combo1=self.findChild(QComboBox,"comboBox")
        self.combo2=self.findChild(QComboBox,"comboBox_2")
        
        self.open1.clicked.connect(self.load_data)
        self.open2.clicked.connect(self.load_data)
        self.out1.clicked.connect(self.Output_data)
        self.out2.clicked.connect(self.Output_data)
        
        self.show()
        
    def load_data(self):
        filepath = QFileDialog.getOpenFileName(self)
        if filepath[0]:
            if self.tab.currentIndex() == 0 :
                self.first_path = filepath[0]
                img = cv2.imread(self.first_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                self.ui.input.show()
                self.ui.input.setImage(np.rot90(img,1))
            elif self.tab.currentIndex() == 1 :
                self.sec_path = filepath[0]
                self.display_img(self.sec_path ,self.inputimg2)
                
    def display_img(self,path,widget):
        self.pixmap = QPixmap(path)
        widget.setScaledContents(True)
        widget.setPixmap(self.pixmap)
    
    def save_plt(self,img,path):
        plt.imshow(img)
        plt.axis('off')
        plt.savefig("./Output/"+path,bbox_inches='tight')
        
    def save_cv(self,img,path):
        cv.imwrite("./Output/"+path,img)
        
    def Output_data(self):
        if self.tab.currentIndex() == 0 :
            img = cv2.imread(self.first_path)
            img = cv2.resize(img,(300,300))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            if self.ui.comboBox.currentText() == "Optimal Local":
                result1 = Local_threshold(img,100,"Optimal")
                self.ui.output.show()
                self.ui.output.setImage(np.rot90(result1,1))
                
            elif self.ui.comboBox.currentText() == "Optimal Global":
                result1 = Global_threshold(img,'Optimal')
                self.ui.output.show()
                self.ui.output.setImage(np.rot90(result1,1))

            elif self.ui.comboBox.currentText() == "Otsu Local":
                result1 = Local_threshold(img,100,"Otsu")
                self.ui.output.show()
                self.ui.output.setImage(np.rot90(result1,1))

            elif self.ui.comboBox.currentText() == "Otsu Global":
                result1 = Global_threshold(img,'Otsu')
                self.ui.output.show()
                self.ui.output.setImage(np.rot90(result1,1))

            elif self.ui.comboBox.currentText() == "Spectral Local":
                result1 = Local_threshold(img,100,"spect")
                self.ui.output.show()
                self.ui.output.setImage(np.rot90(result1,1))

            elif self.ui.comboBox.currentText() == "Spaectral Global":
                result1 = Global_threshold(img,'spect')
                self.ui.output.show()
                self.ui.output.setImage(np.rot90(result1,1))
            else:
                print("err")
        elif self.tab.currentIndex() == 1 :
            
            if self.combo2.currentText() == "Mean Shift":
                image=plt.imread(self.sec_path)
                out_file="meanshift_out.png"
                Bandwidth = 0.1*np.max(image)
                segmented_image, num_clusters = meanShift(image, Bandwidth, 3)
                self.save_plt(segmented_image,out_file)
                self.display_img("./Output/"+out_file,self.outputimg2)
                
            elif self.combo2.currentText() == "Kmeans ":
                image=cv2.imread(self.sec_path)
                img= BGR_To_LUV(image)
                # img = cv2.cvtColor(image, cv.COLOR_BGR2LUV)
                out_file="Kmeans_out.png"
                pixel_values = img.reshape((-1, 3))
                pixel_values = np.float32(pixel_values)
                k = 4
                max_iter = 100
                myModel = KMeans(K=k, max_iterations=max_iter)
                predictions = myModel.predict(pixel_values)
                np.seterr(invalid='ignore')
                centers = np.uint8(myModel.modelCentroids())
                predictions = predictions.astype(int)
                labels = predictions.flatten()
                segmented_image = centers[labels]
                segmented_image = segmented_image.reshape(img.shape)
                self.save_cv(segmented_image,out_file)
                self.display_img("./Output/"+out_file,self.outputimg2)
                # self.display_img("./out/"+out_file,self.outputimg2)
            
            elif self.combo2.currentText() == "Region_Growing":
                image=cv2.imread(self.sec_path,0)
                # image=plt.imread(self.sec_path,0)
                out_file="regGrowing_out.png"
                seeds = [[25, 35],[88, 200],[30, 250]]
                output = fit(image,seeds, 6)
                # self.save_cv(output,out_file)
                self.save_plt(output,out_file)
                self.display_img("./Output/"+out_file,self.outputimg2)
            elif self.combo2.currentText() == "Agglo":
                image=cv2.imread(self.sec_path)
                out_file="agglo_out.png"
                img = np.array(image)
                points,dindogram = segment(img,15)
                output=draw(points,dindogram,image)
                # self.save_cv(output,out_file)
                self.save_cv(output,out_file)
                self.display_img("./Output/"+out_file,self.outputimg2)
            else:
                print("err")
                            
        
app = QApplication(sys.argv)
# application = App()
# application.show()
UIWindow = UI()
app.exec_()