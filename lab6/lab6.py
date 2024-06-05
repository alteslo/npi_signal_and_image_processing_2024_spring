import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import matplotlib.pyplot as plt


class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Обработка изображений")

        self.image = None
        self.noisy_image = None
        self.processed_image = None

        # UI Elements
        self.load_button = ttk.Button(root, text="Загрузить изображение", command=self.load_image)
        self.load_button.pack(pady=10)

        self.noise_button = ttk.Button(root, text="Наложить шум", command=self.add_noise)
        self.noise_button.pack(pady=10)

        self.filter_label = ttk.Label(root, text="Выберите фильтр для удаления шума:")
        self.filter_label.pack()

        self.filter_var = tk.StringVar(value="low")
        self.filter_menu = ttk.Combobox(root, textvariable=self.filter_var)
        self.filter_menu['values'] = ["НЧ фильтр", "ВЧ фильтр", "Медианный фильтр"]
        self.filter_menu.pack(pady=10)

        self.filter_button = ttk.Button(root, text="Применить фильтр", command=self.apply_filter)
        self.filter_button.pack(pady=10)

        self.contrast_label = ttk.Label(root, text="Выберите метод повышения контраста:")
        self.contrast_label.pack()

        self.contrast_var = tk.StringVar(value="Линейная")
        self.contrast_menu = ttk.Combobox(root, textvariable=self.contrast_var)
        self.contrast_menu['values'] = ["Линейная", "Экспоненциальная", "Рэлея", "Степени 2/3", "Гиперболическая"]
        self.contrast_menu.pack(pady=10)

        self.contrast_button = ttk.Button(root, text="Повысить контраст", command=self.enhance_contrast)
        self.contrast_button.pack(pady=10)

        self.image_label = ttk.Label(root)
        self.image_label.pack(pady=10)

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")])
        if file_path:
            self.image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            self.display_image(self.image)

    def display_image(self, img):
        img = Image.fromarray(img)
        img_tk = ImageTk.PhotoImage(image=img)
        self.image_label.img_tk = img_tk
        self.image_label.configure(image=img_tk)

    def add_noise(self):
        if self.image is not None:
            row, col = self.image.shape
            mean = 0
            sigma = 25
            gauss = np.random.normal(mean, sigma, (row, col))
            gauss = gauss.reshape(row, col)
            noisy = self.image + gauss
            self.noisy_image = np.clip(noisy, 0, 255).astype(np.uint8)
            self.display_image(self.noisy_image)

    def apply_filter(self):
        if self.noisy_image is not None:
            filter_type = self.filter_var.get()
            if filter_type == "НЧ фильтр":
                self.processed_image = cv2.GaussianBlur(self.noisy_image, (5, 5), 0)
            elif filter_type == "ВЧ фильтр":
                kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
                self.processed_image = cv2.filter2D(self.noisy_image, -1, kernel)
            elif filter_type == "Медианный фильтр":
                self.processed_image = cv2.medianBlur(self.noisy_image, 5)
            self.display_image(self.processed_image)

    def enhance_contrast(self):
        if self.processed_image is not None:
            method = self.contrast_var.get()
            if method == "Линейная":
                self.processed_image = cv2.equalizeHist(self.processed_image)
            elif method == "Экспоненциальная":
                self.processed_image = self.exponential_contrast(self.processed_image)
            elif method == "Рэлея":
                self.processed_image = self.rayleigh_contrast(self.processed_image)
            elif method == "Степени 2/3":
                self.processed_image = self.gamma_contrast(self.processed_image, gamma=2/3)
            elif method == "Гиперболическая":
                self.processed_image = self.hyperbolic_contrast(self.processed_image)
            self.display_image(self.processed_image)

    def exponential_contrast(self, img):
        img = img / 255.0
        img = np.exp(img) - 1
        img = np.clip(img * 255.0 / np.max(img), 0, 255).astype(np.uint8)
        return img

    def rayleigh_contrast(self, img):
        img = img / 255.0
        img = img ** 2
        img = np.clip(img * 255.0 / np.max(img), 0, 255).astype(np.uint8)
        return img

    def gamma_contrast(self, img, gamma):
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
        return cv2.LUT(img, table)

    def hyperbolic_contrast(self, img):
        img = img / 255.0
        img = np.tanh(img)
        img = np.clip(img * 255.0 / np.max(img), 0, 255).astype(np.uint8)
        return img


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessingApp(root)
    root.mainloop()
