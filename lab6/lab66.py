import tkinter as tk
from tkinter import filedialog, ttk
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure


class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Обработка изображений")

        self.filepath = None
        self.image = None
        self.noisy_image = None
        self.filtered_image = None

        # UI Elements
        self.load_button = ttk.Button(root, text="Загрузить изображение", command=self.load_image)
        self.load_button.pack(pady=10)

        self.noise_button = ttk.Button(root, text="Наложить шум", command=self.add_noise)
        self.noise_button.pack(pady=10)

        self.filter_button = ttk.Button(root, text="Удалить шум", command=self.remove_noise)
        self.filter_button.pack(pady=10)

        self.contrast_button = ttk.Button(root, text="Повысить контраст", command=self.enhance_contrast)
        self.contrast_button.pack(pady=10)

        self.plot_button = ttk.Button(root, text="Показать графики", command=self.plot_images)
        self.plot_button.pack(pady=10)

    def load_image(self):
        self.filepath = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff")])
        if self.filepath:
            self.image = cv2.imread(self.filepath, cv2.IMREAD_GRAYSCALE)
            print(f'Загружено изображение: {self.filepath}')

    def add_noise(self):
        if self.image is None:
            tk.messagebox.showerror("Ошибка", "Пожалуйста, загрузите изображение.")
            return

        noise_type = tk.simpledialog.askstring("Тип шума", "Введите тип шума (gaussian, salt_pepper, uniform):")

        if noise_type == 'gaussian':
            mean = 0
            var = 10
            sigma = var ** 0.5
            gauss = np.random.normal(mean, sigma, self.image.shape)
            self.noisy_image = self.image + gauss
        elif noise_type == 'salt_pepper':
            s_vs_p = 0.5
            amount = 0.04
            noisy = np.copy(self.image)
            num_salt = np.ceil(amount * self.image.size * s_vs_p)
            num_pepper = np.ceil(amount * self.image.size * (1.0 - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_salt)) for i in self.image.shape]
            noisy[coords] = 255
            coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in self.image.shape]
            noisy[coords] = 0
            self.noisy_image = noisy
        elif noise_type == 'uniform':
            noise = np.random.uniform(0, 25, self.image.shape)
            self.noisy_image = self.image + noise

        self.noisy_image = np.clip(self.noisy_image, 0, 255).astype(np.uint8)
        print("Шум добавлен.")

    def remove_noise(self):
        if self.noisy_image is None:
            tk.messagebox.showerror("Ошибка", "Пожалуйста, добавьте шум к изображению.")
            return

        filter_type = tk.simpledialog.askstring("Тип фильтра", "Введите тип фильтра (low, high, median):")

        if filter_type == 'low':
            self.filtered_image = cv2.blur(self.noisy_image, (5, 5))
        elif filter_type == 'high':
            kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
            self.filtered_image = cv2.filter2D(self.noisy_image, -1, kernel)
        elif filter_type == 'median':
            self.filtered_image = cv2.medianBlur(self.noisy_image, 5)

        print("Шум удален.")

    def enhance_contrast(self):
        if self.image is None:
            tk.messagebox.showerror("Ошибка", "Пожалуйста, загрузите изображение.")
            return

        contrast_type = tk.simpledialog.askinteger("Тип контраста", "Введите тип контраста (0: Линейная, 1: Экспоненциальная, 2: Рэлея, 3: Степени 2/3, 4: Гиперболическая):")

        if contrast_type == 0:
            self.image = exposure.rescale_intensity(self.image, in_range='image', out_range='dtype')
        elif contrast_type == 1:
            self.image = exposure.adjust_sigmoid(self.image, cutoff=0.5, gain=10)
        elif contrast_type == 2:
            self.image = exposure.adjust_gamma(self.image, gamma=0.5, gain=1)
        elif contrast_type == 3:
            self.image = exposure.adjust_gamma(self.image, gamma=2/3, gain=1)
        elif contrast_type == 4:
            self.image = exposure.equalize_hist(self.image)

        print("Контраст изображения повышен.")

    def plot_images(self):
        if self.image is None:
            tk.messagebox.showerror("Ошибка", "Пожалуйста, загрузите или смоделируйте сигнал.")
            return

        plt.figure()

        # Исходное изображение
        plt.subplot(2, 2, 1)
        plt.imshow(self.image, cmap='gray')
        plt.title("Исходное изображение")

        # Изображение с шумом
        if self.noisy_image is not None:
            plt.subplot(2, 2, 2)
            plt.imshow(self.noisy_image, cmap='gray')
            plt.title("Изображение с шумом")

        # Фильтрованное изображение
        if self.filtered_image is not None:
            plt.subplot(2, 2, 3)
            plt.imshow(self.filtered_image, cmap='gray')
            plt.title("Фильтрованное изображение")

        # Повышенный контраст
        plt.subplot(2, 2, 4)
        plt.imshow(self.image, cmap='gray')
        plt.title("Изображение с повышенным контрастом")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessingApp(root)
    root.mainloop()
