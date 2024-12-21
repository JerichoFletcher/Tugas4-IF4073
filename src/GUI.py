import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QComboBox, QWidget, QFileDialog
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage

from utils import predict

import cv2

from deep_learning import DeepLearningProcessor

class ImageProcessor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Processor")
        self.resize(800, 600) 

        # UI Components
        self.input_label = QLabel("Input Image")
        self.input_label.setAlignment(Qt.AlignCenter)
        self.output_label = QLabel("Output Image")
        self.output_label.setAlignment(Qt.AlignCenter)

        self.algorithm_selector = QComboBox()
        self.algorithm_selector.addItems(["Conventional", "Deep Learning"])
        self.algorithm_selector.setFixedHeight(40)

        self.load_button = QPushButton("Load Input Image")
        self.load_button.setFixedHeight(40)
        self.process_button = QPushButton("Process Image")
        self.process_button.setFixedHeight(40)

        # Layouts
        image_layout = QHBoxLayout()
        image_layout.addWidget(self.input_label)
        image_layout.addWidget(self.output_label)

        control_layout = QVBoxLayout()
        control_layout.addWidget(self.algorithm_selector)
        control_layout.addWidget(self.load_button)
        control_layout.addWidget(self.process_button)

        main_layout = QVBoxLayout()
        main_layout.addLayout(image_layout)
        main_layout.addLayout(control_layout)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # Connections
        self.load_button.clicked.connect(self.load_input_image)
        self.process_button.clicked.connect(self.process_image)

        # State
        self.input_image = None

        # Modules
        self.module_deep_learning = DeepLearningProcessor()

    def load_input_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.bmp)")
        if file_path:
            self.input_image = cv2.imread(file_path)
            pixmap = QPixmap(file_path)
            self.input_label.setPixmap(pixmap.scaled(
                self.input_label.width(), self.input_label.height(), Qt.KeepAspectRatio
            ))

    def process_image(self):
        if self.input_image is None:
            self.output_label.setText("No input image loaded!")
            return

        selected_algorithm = self.algorithm_selector.currentText()

        if selected_algorithm == "Conventional":
            output_image = self.conventional_processing(self.input_image)
        elif selected_algorithm == "Deep Learning":
            output_image = self.deep_learning_processing(self.input_image)
        else:
            self.output_label.setText("Unknown Algorithm Selected!")
            return

        # Convert output_image to QPixmap and display it
        output_image_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)

        # Convert output_image to QPixmap and display it
        height, width, channel = output_image_rgb.shape
        bytes_per_line = 3 * width
        q_image = QImage(output_image_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.output_label.setPixmap(pixmap.scaled(
            self.output_label.width(), self.output_label.height(), Qt.KeepAspectRatio
        ))

    def conventional_processing(self, image):
        result = predict.predict(image)
        return result

    def deep_learning_processing(self, image):
        result = self.module_deep_learning.process(image)
        return result

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageProcessor()
    window.show()
    sys.exit(app.exec_())
