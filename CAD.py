import cv2
import sys
import os
import numpy as np
import pandas as pd
import pickle
import time
import joblib
from PyQt5 import QtGui, QtCore
from PyQt5.uic import loadUi
from PyQt5.QtWidgets import QApplication, QMainWindow, QGraphicsScene, QFileDialog
from PyQt5.QtGui import QPixmap, QImage, QIcon
from PyQt5.QtCore import Qt, QTimer
from sklearn import preprocessing

if hasattr(sys, '_MEIPASS'):
    base_path = sys._MEIPASS
    ui_path = os.path.join(sys._MEIPASS, 'CAD.ui')
else:
    base_path = os.path.abspath(".")
    ui_path = 'CAD.ui'

model_svm_path = os.path.join(base_path, "model", "ModelSVM")
model_svm_pkl_path = os.path.join(base_path, "model", "ModelSVM.pkl")
tumourfeature_csv_path = os.path.join(base_path, "model", "tumourfeature.csv")
logo_pens_path = os.path.join(base_path, "asset", "Logo_PENS_putih.png")
logo_jpg_path = os.path.join(base_path, "asset", "logo.png")

class CLA: 
    def __init__(self,rows,cols,initial_state=None):
        self.rows = rows
        self.cols = cols
        if initial_state is None:
            self.state = np.zeros((rows, cols), dtype=np.uint8)
        else:
            self.state = initial_state
        self.learning_rate = 0.1

    def get_neighborhood(self, i, j):
        neighborhood = self.state[i - 1:i + 2, j - 1:j + 2]
        return neighborhood.flatten()
    
    def update_cell(self, i, j): 
        neighborhood = self.get_neighborhood(i, j)
        current_state = self.state[i,j]
        if np.sum(neighborhood) >= 5:
            self.state[i,j] = 1
        else:
            self.state[i,j] = 0
        if self.state[i,j] != current_state:
            self.learn(neighborhood)

    def learn(self, neighborhood):
        for i in range(len(neighborhood)):
            if neighborhood[i] == 1:
                self.learning_rate = max(0.1, self.learning_rate - 0.01)
            else:
                self.learning_rate = min(0.9, self.learning_rate + 0.01)

    def iteration(self, num_iteration):
        for iteration in range(num_iteration):
            start_time = time.time()
            for i in range(1, self.rows - 1):
                for j in range (1, self.cols - 1):
                    self.update_cell(i, j)
            end_time = time.time()
            duration = end_time - start_time
            print(f"Time consumed by iteration {iteration + 1}: {duration:.2f} seconds")
        
class MainUI(QMainWindow):
    def __init__(self):
        super(MainUI, self).__init__()
        loadUi(ui_path, self)

        self.setWindowIcon(QIcon(logo_jpg_path))

        self.selected_contour = []
        self.selected_contour_1 = []
        self.selected_contour_2 = []
        self.binary_image = None
        self.noduleData = None
        self.original_preprocessed = None
        self.tumour_class = None

        self.pushButton.clicked.connect(self.load_image)
        self.pushButton_2.clicked.connect(self.process_image) 
        self.pushButton_3.clicked.connect(self.draw_image)
        self.pushButton_4.clicked.connect(self.segmented_image)
        self.pushButton_5.clicked.connect(self.extract_image)
        self.pushButton_7.clicked.connect(self.perform_detection) 
        self.pushButton_8.clicked.connect(self.clear_process) 
        self.pushButton_9.clicked.connect(self.save_diagnose) 
        self.pushButton_10.clicked.connect(self.svm_classify) 

        self.scene = self.graphicsView.scene()
        if self.scene is None:
            self.scene = QGraphicsScene()
            self.graphicsView.setScene(self.scene)
        
        self.scene_2 = self.graphicsView_2.scene()
        if self.scene_2 is None:
            self.scene_2 = QGraphicsScene()
            self.graphicsView_2.setScene(self.scene_2)

        self.scene_3 = self.graphicsView_3.scene()
        if self.scene_3 is None:
            self.scene_3 = QGraphicsScene()
            self.graphicsView_3.setScene(self.scene_3)
        
    def load_image(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg)")
        if not filepath:
            return
        scan = cv2.imread(filepath)
        width=512
        height=512
        image = scan

        orig_height, orig_width = image.shape[:2]
        aspect_ratio = orig_width / orig_height

        if width / height > aspect_ratio:
            new_width = int(height * aspect_ratio)
            ew_height = height
        else:
            new_width = width
            new_height = int(width / aspect_ratio)
        
        resized_image = cv2.resize(image, (new_width, new_height))
        canvas = 255 * np.ones((height, width, 3), dtype=np.uint8)
        
        x_offset = (width - new_width) // 2
        y_offset = (height - new_height) // 2
        
        if len(resized_image.shape) == 3 and resized_image.shape[2] == 4:
            resized_image = cv2.cvtColor(resized_image, cv2.COLOR_RGBA2RGB)
        
        canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_image
        scan=canvas
        self.scan=scan
        scan = cv2.cvtColor(scan, cv2.COLOR_BGR2GRAY)

        q_image = QImage(scan.data, scan.shape[1], scan.shape[0], QImage.Format_Grayscale8)

        pixmap = QPixmap.fromImage(q_image)
        if not pixmap.isNull():
            scene = self.graphicsView.scene()
            scene.clear()
            scene.addPixmap(pixmap)
            self.graphicsView.fitInView(scene.sceneRect(), Qt.AspectRatioMode.IgnoreAspectRatio)

    def process_image(self):
        items = self.scene.items()
        if not items:
            return 
        image = self.scene.items()[0].pixmap().toImage()
        image = image.convertToFormat(QImage.Format_Grayscale8)
        image_bits = image.bits().asstring(image.byteCount())
        image_data = np.frombuffer(image_bits, dtype=np.uint8)
        height, width = image.height(), image.width()
        image_data = image_data.reshape(height, width) 
        
        gabor_kernel = cv2.getGaborKernel((50,50), theta=0, psi=0, lambd=5, sigma=1, gamma=1)
        image_enhance = cv2.filter2D(image_data, cv2.CV_8UC1, gabor_kernel)
        img_norm = cv2.normalize(image_enhance, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        self.original_preprocessed = cv2.cvtColor(img_norm, cv2.COLOR_GRAY2RGB)

        processed_image = QtGui.QImage(img_norm.data, img_norm.shape[1], 
                                       img_norm.shape[0], QtGui.QImage.Format_Grayscale8)

        pixmap = QtGui.QPixmap.fromImage(processed_image)
        if not pixmap.isNull():
            self.scene.clear()
            self.scene.addPixmap(pixmap)
            self.graphicsView.fitInView(self.scene.sceneRect(), QtCore.Qt.AspectRatioMode.IgnoreAspectRatio)

    def draw_image(self):
        items = self.scene.items()
        if not items or self.scene is None:
            return
        if not hasattr(self, 'original_preprocessed') or self.original_preprocessed is None:
            self.result_text.setPlainText("Please complete the process before!")
            self.detect_text.setPlainText("Error processing image!")
            self.cell_text.setPlainText("Error processing image!")
            return

        pixmap = self.scene.items()[0].pixmap()
        if not pixmap.isNull():
            image = pixmap.toImage()
            image = image.convertToFormat(QtGui.QImage.Format_RGB888)
            buffer = image.bits().asstring(image.byteCount())
            img_data = np.frombuffer(buffer, dtype=np.uint8).reshape(image.height(), image.width(), 3)

            highlight = img_data.copy()
            self.highlight = highlight

            gray = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            contours = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            contours = contours[0] if len(contours) == 2 else contours[1]
            self.contours = contours
            fixed_contour = []
            for c in contours:
                area = cv2.contourArea(c)
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.035 * peri, True)
                (x, y, w, h) = cv2.boundingRect(approx)
                aspect_ratio = w / float(h)
                if aspect_ratio <= 1.2 or area < 5000:
                    fixed_contour.append(c) 
            if len(fixed_contour) >= 2: 
                sorted_contour = sorted(fixed_contour, key=cv2.contourArea, reverse=True) 
                first_contour = sorted_contour[0] 
                second_contour = sorted_contour[1]
                cv2.drawContours(self.highlight, [first_contour], 0, (0,255,0), 2)
                cv2.drawContours(self.highlight, [second_contour], 0, (0,255,0), 2)
                self.selected_contour_1 = [first_contour, second_contour] 
                self.selected_contour = self.selected_contour_1

        result_image = QtGui.QImage(self.highlight.data, self.highlight.shape[1], 
                                        self.highlight.shape[0], self.highlight.shape[1] * 3, QtGui.QImage.Format_RGB888)

        result_pixmap = QtGui.QPixmap.fromImage(result_image)
        if not result_pixmap.isNull():
            self.scene_2.clear()
            self.scene_2.addPixmap(result_pixmap)
            self.graphicsView_2.fitInView(self.scene_2.sceneRect(), QtCore.Qt.AspectRatioMode.IgnoreAspectRatio)
            self.graphicsView_2.mousePressEvent = self.mouse_click

    def mouse_click(self, event):
        items = self.scene.items()
        if not items:
            return 
        if event.button() == Qt.LeftButton:
            x,y = event.pos().x(),event.pos().y()
            selected_contour_index = None
            for i,c in enumerate(self.contours):
                r = cv2.pointPolygonTest(c, (x, y), False)
                if r >= 0:
                    selected_contour_index = i
                    break
            if selected_contour_index is not None:
                selected_contour = self.contours[selected_contour_index]
                self.selected_contour_2.append(selected_contour)
                for contour in self.selected_contour_2:
                    cv2.drawContours(self.highlight, [contour], 0, (255,0,0), 2)
            self.selected_contour = []
            self.selected_contour = self.selected_contour_2
            
        mouse_click = QtGui.QImage(self.highlight.data, self.highlight.shape[1], 
                                       self.highlight.shape[0], self.highlight.shape[1] * 3, QtGui.QImage.Format_RGB888)

        result_pixmap = QtGui.QPixmap.fromImage(mouse_click)
        if not result_pixmap.isNull():
            self.scene_2.clear()
            self.scene_2.addPixmap(result_pixmap)
            self.graphicsView_2.fitInView(self.scene_2.sceneRect(), QtCore.Qt.AspectRatioMode.IgnoreAspectRatio)

    def segmented_image(self):
        items = self.scene.items()
        if not items:
            return 
        if not hasattr(self, 'original_preprocessed') or self.original_preprocessed is None:
            self.result_text.setPlainText("Please complete the process before!")
            self.detect_text.setPlainText("Error processing image!")
            self.cell_text.setPlainText("Error processing image!")
            return
        if not self.selected_contour:
            return
        mask = np.zeros_like(self.highlight)
        for contour in self.selected_contour:
            cv2.drawContours(mask, [contour], 0, (255,255,255), -1)
        self.lungsize = np.sum(mask==255)
        try:
            result = cv2.bitwise_and(self.original_preprocessed, mask)
            self.result = result
        except Exception:
            self.result_text.setPlainText("Please complete the process before!")
            self.detect_text.setPlainText("Error processing image!")
            self.cell_text.setPlainText("Error processing image!")
            return

        result_segmented = QtGui.QImage(result.data, result.shape[1], result.shape[0], result.shape[1] * 3, QtGui.QImage.Format_RGB888)
        
        pixmap = QtGui.QPixmap.fromImage(result_segmented)
        if not pixmap.isNull():
            self.scene_2.clear()
            self.scene_2.addPixmap(pixmap)
            self.graphicsView_2.fitInView(self.scene_2.sceneRect(), QtCore.Qt.AspectRatioMode.IgnoreAspectRatio)
    
    def extract_image(self):
        items = self.scene.items()
        if not items:
            return 
        if not hasattr(self, 'result') or self.result is None:
            self.result_text.setPlainText("Please complete the process before!")
            self.detect_text.setPlainText("Error processing image!")
            self.cell_text.setPlainText("Error processing image!")
            return
        try:
            gray = cv2.cvtColor(self.result.copy(), cv2.COLOR_BGR2GRAY)
            _, binary_image = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        except Exception:
            self.result_text.setPlainText("Please complete the process before!")
            self.detect_text.setPlainText("Error processing image!")
            self.cell_text.setPlainText("Error processing image!")
            return
        self.binary_image = binary_image

        extract_image = QtGui.QImage(binary_image.data, binary_image.shape[1], 
                                     binary_image.shape[0], QtGui.QImage.Format_Grayscale8)
    
        pixmap = QtGui.QPixmap.fromImage(extract_image)
        if not pixmap.isNull():
            self.scene_3.clear()
            self.scene_3.addPixmap(pixmap)
            self.graphicsView_3.fitInView(self.scene_3.sceneRect(), QtCore.Qt.AspectRatioMode.IgnoreAspectRatio)

    def perform_detection(self):
        items = self.scene.items()
        if not items or self.binary_image is None:
            self.result_text.setPlainText(str("Please complete the process!"))
            self.detect_text.setPlainText(str("Image not available!"))
            self.cell_text.setPlainText(str("Image not available!"))
            return 
        self.result_text.setPlainText(str("Diagnosis in progress, please don't close..."))
        self.detect_text.setPlainText(str("Detection in progress..."))
        self.cell_text.setPlainText(str("Processing data...."))
        QTimer.singleShot(500, self.detection_image)
        
    def detection_image(self):
        items = self.scene.items()
        if not items or self.binary_image is None:
            return 
        try:
            binary_image = self.binary_image.copy() 
            automata = CLA(rows=binary_image.shape[0], cols=binary_image.shape[1], initial_state=binary_image)
        except Exception:
            self.result_text.setPlainText("Please complete the process!")
            self.detect_text.setPlainText("Error processing image!")
            self.cell_text.setPlainText("Error processing image!")
            return
        start_time = time.time()
        automata.iteration(num_iteration=10)
        end_time = time.time()
        duration = end_time - start_time
        print(f"Time consumed by CA: {duration:.2f} seconds")
        result=automata.state
        self.plot_active_cells(result)
        
        tumour_count = np.sum(result)
        total_count = self.lungsize
        tumour_posibility = (tumour_count / total_count) * 100
        if tumour_posibility > 0.3:
            self.detect_text.setPlainText(str("Abnormal lung"))
            self.svm_contour()
            self.cell_text.setPlainText(str("{:.2f}%".format(tumour_posibility)))
        else:
            self.tumour_class = "Normal lung"
            self.detect_text.setPlainText(str("Normal lung"))
            self.result_text.setPlainText(str("Patient is Healthy"))
            self.cell_text.setPlainText(str("{:.2f}% (non tumour)".format(tumour_posibility)))
            return
    
    def plot_active_cells(self, result):
        items = self.scene.items()
        if not items or self.result is None:
            return 
        result_image = self.result.copy()
        cancer_cells = np.argwhere(result == 1) 
        for cell in cancer_cells: 
            cv2.drawMarker(result_image, (cell[1], cell[0]), (0, 255, 0), markerType=cv2.MARKER_SQUARE, markerSize=1, thickness=1)

        result_image = QtGui.QImage(result_image.data, result_image.shape[1], result_image.shape[0], 
                                    result_image.shape[1] * 3, QtGui.QImage.Format_RGB888)

        result_pixmap = QtGui.QPixmap.fromImage(result_image)
        if not result_pixmap.isNull():
            self.scene_2.clear()
            self.scene_2.addPixmap(result_pixmap)
            self.graphicsView_2.fitInView(self.scene_2.sceneRect(), QtCore.Qt.AspectRatioMode.IgnoreAspectRatio)
    
    def svm_contour(self):
        items = self.scene.items()
        if not items or self.binary_image is None:
            return 
        noduleData = []
        features = []
        binary_image = self.binary_image
        binary_image = binary_image.copy()
        original = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2RGB)
        contours, _ = cv2.findContours(binary_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        self.svm_contours = contours
        features = list(features)
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
        biggest_contour = sorted_contours[0]
        area = cv2.contourArea(biggest_contour)
        peri = cv2.arcLength(biggest_contour, True)
        if peri!=0:
            roundness = (4*3.14*area) / (peri*peri)      
            features.append([area, peri, roundness])
            cv2.drawContours(original, [biggest_contour], 0, (0,0,255), 2)
            
        result_image = QtGui.QImage(original.data, original.shape[1], original.shape[0], 
                                    original.shape[1]*3, QtGui.QImage.Format_RGB888)

        result_pixmap = QtGui.QPixmap.fromImage(result_image)
        if not result_pixmap.isNull():
            self.scene_3.clear()
            self.scene_3.addPixmap(result_pixmap)
            self.graphicsView_3.fitInView(self.scene_3.sceneRect(), QtCore.Qt.AspectRatioMode.IgnoreAspectRatio)
            self.graphicsView_3.mousePressEvent = self.click_event_svm

        noduleData = features
        self.noduleData = np.array(noduleData, dtype=np.float32)
        features = []

    def click_event_svm(self, event):
        items = self.scene.items()
        if not items or self.binary_image is None:
            return 
        if self.binary_image is None:
            return
        binary = self.binary_image
        binary = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
        if event.button() == Qt.LeftButton:
            x, y = event.pos().x(), event.pos().y()
            contours = self.svm_contours
            selected_contour_index = None

            for i, c in enumerate(contours):
                r = cv2.pointPolygonTest(c, (x, y), False)
                if r >= 0:
                    selected_contour_index = i
                    break

            if selected_contour_index is not None:
                selected_contour = contours[selected_contour_index]
                cv2.drawContours(binary, [selected_contour], 0, (0, 255, 0), 2)
            
                features = []
                noduleData = []
                area = cv2.contourArea(selected_contour)
                peri = cv2.arcLength(selected_contour, True)
                if peri!=0:
                    roundness = (4*3.14*area) / (peri*peri)

                    features.append([area, peri, roundness])

                noduleData = features
                self.noduleData = np.array(noduleData, dtype=np.float32)
                features = [] 

        contour_svm = QtGui.QImage(binary.data, binary.shape[1], binary.shape[0],
                                               binary.shape[1] * 3, QtGui.QImage.Format_RGB888)

        contour_svm_pixmap = QtGui.QPixmap.fromImage(contour_svm)
        if not contour_svm_pixmap.isNull():
            self.scene_3.clear()
            self.scene_3.addPixmap(contour_svm_pixmap)
            self.graphicsView_3.fitInView(self.scene_3.sceneRect(), QtCore.Qt.AspectRatioMode.IgnoreAspectRatio)

    def svm_classify(self):
        items = self.scene.items()
        if not items or self.binary_image is None or self.noduleData is None:
            self.result_text.setPlainText(str("Please complete the process!"))
            self.detect_text.setPlainText(str("Image not available!"))
            self.cell_text.setPlainText(str("Image not available!"))
            return 
        try:
            area = self.noduleData[0][0]
            peri = self.noduleData[0][1]
            roundness = self.noduleData[0][2]
        except Exception:
            return
        
        with open("model/tumourfeature.csv", "a") as myfile:
           myfile.write(f"{str(area)},{str(peri)},{str(roundness)}\n")
        
        df = pd.read_csv('model/tumourfeature.csv')
        X = df.drop('CLASS', axis=1)
        X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
        temp = X.shape[0]
        filename = 'model/ModelSVM'
        loaded_model = joblib.load(filename)
        model = pickle.dumps(loaded_model)
        prediction = pickle.loads(model)
        predict = prediction.predict(X[temp-1:temp])
        
        with open("model/tumourfeature.csv", "r") as data:
            lines = data.readlines()
            lines = lines[:-1]
        with open("model/tumourfeature.csv", "w") as data:
            for line in lines:
                data.write(line)

        if int(predict[0] == 0):
            self.tumour_class = "Benign tumour"
            self.result_text.setPlainText(str("Benign tumour"))
        elif int(predict[0] == 1):
            self.tumour_class = "Malignant tumour"
            self.result_text.setPlainText(str("Malignant tumour"))

    def save_diagnose(self):
        items = self.scene.items()
        if not items or self.scan is None or (not self.tumour_class or len(self.tumour_class) == 0):
            self.result_text.setPlainText(str("Please complete the diagnosis process!"))
            self.detect_text.setPlainText(str("Error saving image!"))
            self.cell_text.setPlainText(str("Error saving image!"))
            return
        image_to_save = self.scan.copy()
        diagnosis_text = f'Diagnosis: {self.tumour_class}'
        cv2.putText(image_to_save, diagnosis_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        filename, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "Images (*.png *.jpg *.jpeg)")
        if filename:
            cv2.imwrite(filename, cv2.cvtColor(image_to_save, cv2.COLOR_RGB2BGR))
            print(f"Image saved as {filename}")
        else:
            print("Save cancelled.")

    def clear_process(self):
        self.selected_contour = []
        self.selected_contour_1 = []
        self.selected_contour_2 = []
        self.binary_image = []
        self.noduleData =[]
        self.result = []
        self.original_preprocessed = []
        self.tumour_class = []
        self.scene.clear()
        self.scene_2.clear()
        self.scene_3.clear()
        self.detect_text.clear()
        self.cell_text.clear()
        self.result_text.clear()
    
def main():
    app = QApplication(sys.argv)
    window = MainUI()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()