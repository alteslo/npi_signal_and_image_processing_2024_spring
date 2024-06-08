import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk


class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Обработка изображений")

        self.image = None
        self.processed_image = None

        # UI Elements
        self.load_button = ttk.Button(root, text="Загрузить изображение", command=self.load_image)
        self.load_button.pack(pady=10)

        self.save_button = ttk.Button(root, text="Сохранить изображение", command=self.save_image)
        self.save_button.pack(pady=10)

        self.convert_gray_button = ttk.Button(root, text="Цветное -> Полутоновое", command=self.convert_to_gray)
        self.convert_gray_button.pack(pady=10)

        self.convert_binary_button = ttk.Button(root, text="Полутоновое -> Бинарное", command=self.convert_to_binary)
        self.convert_binary_button.pack(pady=10)

        self.negative_button = ttk.Button(root, text="Негатив", command=self.convert_to_negative)
        self.negative_button.pack(pady=10)

        self.log_transform_button = ttk.Button(root, text="Логарифмическое преобразование", command=self.log_transform)
        self.log_transform_button.pack(pady=10)

        self.power_transform_gt1_button = ttk.Button(root, text="Степенное преобразование (y > 1)", command=lambda: self.power_transform(2))
        self.power_transform_gt1_button.pack(pady=10)

        self.power_transform_lt1_button = ttk.Button(root, text="Степенное преобразование (y < 1)", command=lambda: self.power_transform(0.5))
        self.power_transform_lt1_button.pack(pady=10)

        self.image_label = ttk.Label(root)
        self.image_label.pack(pady=10)

    def load_image(self):
        filetypes = [
            ("BMP files", "*.bmp"),
            ("JPEG files", "*.jpg;*.jpeg"),
            ("PNG files", "*.png"),
        ]
        filepath = filedialog.askopenfilename(filetypes=filetypes)
        if filepath:
            self.image = cv2.imread(filepath)
            if self.image is None:
                messagebox.showerror("Ошибка", "Не удалось загрузить изображение. Пожалуйста, убедитесь, что файл существует и имеет корректный формат.")
            else:
                self.display_image(self.image)
                print(f'Загружено изображение: {filepath}')

    def save_image(self):
        if self.processed_image is None:
            messagebox.showerror("Ошибка", "Нет обработанного изображения для сохранения.")
            return

        filetypes = [
            ("BMP files", "*.bmp"),
            ("JPEG files", "*.jpg;*.jpeg"),
            ("PNG files", "*.png"),
        ]
        filepath = filedialog.asksaveasfilename(filetypes=filetypes)
        if filepath:
            cv2.imwrite(filepath, self.processed_image)
            print(f'Изображение сохранено: {filepath}')

    def convert_to_gray(self):
        if self.image is None:
            messagebox.showerror("Ошибка", "Пожалуйста, загрузите изображение.")
            return
        self.processed_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.display_image(self.processed_image)

    def convert_to_binary(self):
        if self.processed_image is None:
            messagebox.showerror("Ошибка", "Пожалуйста, преобразуйте изображение в полутоновое.")
            return
        _, self.processed_image = cv2.threshold(self.processed_image, 127, 255, cv2.THRESH_BINARY)
        self.display_image(self.processed_image)

    def convert_to_negative(self):
        if self.image is None:
            messagebox.showerror("Ошибка", "Пожалуйста, загрузите изображение.")
            return
        self.processed_image = cv2.bitwise_not(self.image)
        self.display_image(self.processed_image)

    def log_transform(self):
        if self.image is None:
            messagebox.showerror("Ошибка", "Пожалуйста, загрузите изображение.")
            return
        c = 255 / np.log(1 + np.max(self.image))
        image_log = np.where(self.image == 0, 1, self.image)  # Заменяем нули на единицы, чтобы избежать деления на ноль
        self.processed_image = (c * (np.log(1 + image_log))).astype(np.uint8)
        self.display_image(self.processed_image)

    def power_transform(self, gamma):
        if self.image is None:
            messagebox.showerror("Ошибка", "Пожалуйста, загрузите изображение.")
            return
        self.processed_image = np.array(255 * (self.image / 255) ** gamma, dtype=np.uint8)
        self.display_image(self.processed_image)

    def display_image(self, image):
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image = Image.fromarray(image)
        image_tk = ImageTk.PhotoImage(image)
        self.image_label.config(image=image_tk)
        self.image_label.image = image_tk


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessingApp(root)
    root.mainloop()
