import tkinter as tk
from tkinter import Toplevel, filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk


class ImageProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processor")
        self.root.geometry("800x600")

        self.image = None
        self.processed_image = None

        self.load_button = tk.Button(root, text="Load Image", command=self.load_image)
        self.load_button.pack()

        self.noise_button = tk.Button(root, text="Add Noise", command=self.add_noise)
        self.noise_button.pack()

        self.low_pass_button = tk.Button(root, text="Low-Pass Filter", command=self.low_pass_filter)
        self.low_pass_button.pack()

        self.high_pass_button = tk.Button(root, text="High-Pass Filter", command=self.high_pass_filter)
        self.high_pass_button.pack()

        self.median_button = tk.Button(root, text="Median Filter", command=self.median_filter)
        self.median_button.pack()

        self.contrast_button = tk.Button(root, text="Enhance Contrast", command=self.enhance_contrast)
        self.contrast_button.pack()

        self.image_label = tk.Label(root)
        self.image_label.pack()

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image = cv2.imread(file_path)
            self.display_image(self.image)

    def add_noise(self):
        if self.image is None:
            messagebox.showwarning("Warning", "Please load an image first")
            return
        noisy_image = self.image + np.random.normal(loc=0, scale=25, size=self.image.shape)
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        self.processed_image = noisy_image
        self.display_image(noisy_image)

    def low_pass_filter(self):
        if self.processed_image is None:
            messagebox.showwarning("Warning", "Please add noise to the image first")
            return

        low_pass_kernels = [
            np.ones((3, 3), np.float32) / 9,
            np.ones((5, 5), np.float32) / 25,
            np.ones((7, 7), np.float32) / 49
        ]
        name_kernel_dict = {0: "3x3 ядро", 1: "5x5 ядро", 2: "7x7 ядро"}
        for i, kernel in enumerate(low_pass_kernels):
            denoised = cv2.filter2D(self.processed_image, -1, kernel)
            self.show_in_new_window(denoised, title=f"Low-Pass Filter {name_kernel_dict[i]}")

    def high_pass_filter(self):
        if self.processed_image is None:
            messagebox.showwarning("Warning", "Please add noise to the image first")
            return

        high_pass_kernels = [
            np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
            np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),
            np.array([[1, -2, 1], [-2, 5, -2], [1, -2, 1]])
        ]
        name_kernel_dict = {
            0: "[[0, -1, 0], [-1, 5, -1], [0, -1, 0]]",
            1: "[[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]",
            2: "[[1, -2, 1], [-2, 5, -2], [1, -2, 1]]"
        }
        for i, kernel in enumerate(high_pass_kernels):
            denoised = cv2.filter2D(self.processed_image, -1, kernel)
            self.show_in_new_window(denoised, title=f"High-Pass Filter {name_kernel_dict[i]}")

    def median_filter(self):
        if self.processed_image is None:
            messagebox.showwarning("Warning", "Please add noise to the image first")
            return

        median_blurs = [
            cv2.medianBlur(self.processed_image, 3),
            cv2.medianBlur(self.processed_image, 5),
            cv2.medianBlur(self.processed_image, 7)
        ]
        name_kernel_dict = {0: "3x3 ядро", 1: "5x5 ядро", 2: "7x7 ядро"}
        for i, denoised in enumerate(median_blurs):
            self.show_in_new_window(denoised, title=f"Median Filter {name_kernel_dict[i]}")

    def enhance_contrast(self):
        if self.image is None:
            messagebox.showwarning("Warning", "Please load an image first")
            return

        enhanced_images = []

        # Linear
        linear_image = cv2.convertScaleAbs(self.image, alpha=1.5, beta=0)
        enhanced_images.append(linear_image)

        # Exponential
        exp_image = cv2.convertScaleAbs(self.image)
        exp_image = np.uint8(255 * (exp_image / 255) ** 2)
        enhanced_images.append(exp_image)

        # Rayleigh
        rayleigh_image = cv2.convertScaleAbs(self.image)
        rayleigh_image = np.uint8(255 * (rayleigh_image / 255) ** 1.5)
        enhanced_images.append(rayleigh_image)

        # Power 2/3
        power_image = cv2.convertScaleAbs(self.image)
        power_image = np.uint8(255 * (power_image / 255) ** (2/3))
        enhanced_images.append(power_image)

        # Hyperbolic
        hyperbolic_image = cv2.convertScaleAbs(self.image)
        hyperbolic_image = np.uint8(255 * (np.tanh(hyperbolic_image / 255) * (2 - np.tanh(hyperbolic_image / 255))))
        enhanced_images.append(hyperbolic_image)

        name_kernel_dict = {
            0: "Linear",
            1: "Exponential",
            2: "Rayleigh",
            3: "Power 2/3",
            4: "Hyperbolic",
        }
        for i, img in enumerate(enhanced_images):
            self.show_in_new_window(img, title=f"Enhanced Contrast {name_kernel_dict[i]}")

    def display_image(self, img, title="Processed Image"):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(image=img_pil)

        self.image_label.configure(image=img_tk)
        self.image_label.image = img_tk
        self.root.title(title)

    def show_in_new_window(self, img, title="Image"):
        new_window = Toplevel(self.root)
        new_window.title(title)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(image=img_pil)

        label = tk.Label(new_window, image=img_tk)
        label.image = img_tk
        label.pack()


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()
