# -*- coding: utf-8 -*-
import numpy as np
import cv2
import matplotlib.pyplot as plt
from google.colab import files
from scipy import ndimage
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 10

def load_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_color = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    return img_color, img_gray

def display_images(images, titles, rows=2, cols=3, cmap='gray'):
    fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*4))
    axes = axes.ravel()

    for idx, (img, title) in enumerate(zip(images, titles)):
        if img is None:
            axes[idx].axis('off')
            continue

        if len(img.shape) == 3:
            axes[idx].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            axes[idx].imshow(img, cmap=cmap)

        axes[idx].set_title(title, fontsize=12, fontweight='bold')
        axes[idx].axis('off')

    plt.tight_layout()
    plt.show()

def normalize_image(img):
    img_normalized = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(img_normalized)

def apply_mean_filter(img, kernel_size=5):
    return cv2.blur(img, (kernel_size, kernel_size))

def apply_gaussian_filter(img, kernel_size=5, sigma=1.0):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)

def apply_median_filter(img, kernel_size=5):
    return cv2.medianBlur(img, kernel_size)

def apply_bilateral_filter(img, d=9, sigma_color=75, sigma_space=75):
    return cv2.bilateralFilter(img, d, sigma_color, sigma_space)

def apply_laplacian_filter(img):
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    return normalize_image(np.absolute(laplacian))

def apply_sobel_filter(img):
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = np.sqrt(sobel_x**2 + sobel_y**2)
    return normalize_image(sobel_combined)

def apply_prewitt_filter(img):
    kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    prewitt_x = cv2.filter2D(img.astype(float), -1, kernel_x)
    prewitt_y = cv2.filter2D(img.astype(float), -1, kernel_y)
    prewitt_combined = np.sqrt(prewitt_x**2 + prewitt_y**2)
    return normalize_image(prewitt_combined)

def apply_canny_edge(img, threshold1=50, threshold2=150):
    return cv2.Canny(img, threshold1, threshold2)

def apply_custom_sharpen(img):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)

def process_single_image(img_color, img_gray, image_name):
    mean_filtered = apply_mean_filter(img_gray, kernel_size=5)
    gaussian_filtered = apply_gaussian_filter(img_gray, kernel_size=5, sigma=1.0)
    median_filtered = apply_median_filter(img_gray, kernel_size=5)
    bilateral_filtered = apply_bilateral_filter(img_gray, d=9, sigma_color=75, sigma_space=75)

    images_low = [
        img_gray, mean_filtered, gaussian_filtered,
        median_filtered, bilateral_filtered, None
    ]
    titles_low = [
        f'Original\n{image_name}',
        'Mean Filter',
        'Gaussian Filter',
        'Median Filter',
        'Bilateral Filter',
        None
    ]
    display_images(images_low, titles_low, rows=2, cols=3)

    laplacian = apply_laplacian_filter(img_gray)
    sobel = apply_sobel_filter(img_gray)
    prewitt = apply_prewitt_filter(img_gray)
    canny = apply_canny_edge(img_gray, threshold1=50, threshold2=150)
    sharpened = apply_custom_sharpen(img_gray)

    images_high = [
        img_gray, laplacian, sobel,
        prewitt, canny, sharpened
    ]
    titles_high = [
        f'Original\n{image_name}',
        'Laplacian Filter',
        'Sobel Filter',
        'Prewitt Filter',
        'Canny Edge Detection',
        'Custom Sharpen'
    ]
    display_images(images_high, titles_high, rows=2, cols=3)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    filters_low = [img_gray, mean_filtered, gaussian_filtered, bilateral_filtered]
    titles_compare_low = ['Original', 'Mean', 'Gaussian', 'Bilateral']

    for ax, img, title in zip(axes, filters_low, titles_compare_low):
        ax.imshow(img, cmap='gray')
        ax.set_title(f'{title}\n{image_name}', fontweight='bold')
        ax.axis('off')
    plt.suptitle('Low-Pass Filters Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    filters_high = [img_gray, laplacian, sobel, prewitt, canny]
    titles_compare_high = ['Original', 'Laplacian', 'Sobel', 'Prewitt', 'Canny']

    for ax, img, title in zip(axes, filters_high, titles_compare_high):
        ax.imshow(img, cmap='gray')
        ax.set_title(f'{title}\n{image_name}', fontweight='bold')
        ax.axis('off')
    plt.suptitle('High-Pass Filters Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

    return {
        'low_pass': {
            'mean': mean_filtered,
            'gaussian': gaussian_filtered,
            'median': median_filtered,
            'bilateral': bilateral_filtered
        },
        'high_pass': {
            'laplacian': laplacian,
            'sobel': sobel,
            'prewitt': prewitt,
            'canny': canny,
            'sharpened': sharpened
        }
    }

def main():
    uploaded = files.upload()

    if len(uploaded) < 2:
        print("Can it nhat 2 anh!")
        return

    all_results = {}

    for idx, (filename, content) in enumerate(uploaded.items(), 1):
        from io import BytesIO
        file_obj = BytesIO(content)
        img_color, img_gray = load_image(file_obj)
        results = process_single_image(img_color, img_gray, filename)
        all_results[filename] = results

if __name__ == "__main__":
    main()