# PHAN 1 : RGB TO GRAYSCALE
import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files
from PIL import Image
import io

plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 10

def upload_images():
    uploaded = files.upload()
    images = {}
    for filename in uploaded.keys():
        image_data = uploaded[filename]
        # Doc anh dang BGR roi chuyen sang RGB
        img = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images[filename] = img_rgb
        print(f"Da tai anh: {filename} - Kich thuoc: {img_rgb.shape}")
    return images

# Upload va chon anh de xu ly
images_dict = upload_images()
image_names = list(images_dict.keys())
original_image = images_dict[image_names[0]]

print(f"\nImage name: {image_names[0]}")
print(f"Size: {original_image.shape[0]}x{original_image.shape[1]} pixels")

# CHUYEN DOI ANH MAU SANG ANH XAM

def rgb_to_grayscale_manual(img):
    # Cong thuc chuan: Gray = 0.299*R + 0.587*G + 0.114*B
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]
    gray = 0.299 * R + 0.587 * G + 0.114 * B
    return gray.astype(np.uint8)

def rgb_to_grayscale_average(img):
    # Phuong phap trung binh: Gray = (R+G+B)/3
    return np.mean(img, axis=2).astype(np.uint8)

# Ap dung 3 phuong phap chuyen xam
gray_standard = rgb_to_grayscale_manual(original_image)
gray_average = rgb_to_grayscale_average(original_image)
gray_opencv = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)

# Hien thi ket qua
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

axes[0, 0].imshow(original_image)
axes[0, 0].set_title('Anh mau goc (RGB)', fontsize=14, fontweight='bold')
axes[0, 0].axis('off')

axes[0, 1].imshow(gray_standard, cmap='gray')
axes[0, 1].set_title('Anh xam - Phuong phap chuan\n(0.299R + 0.587G + 0.114B)',
                      fontsize=14, fontweight='bold')
axes[0, 1].axis('off')

axes[1, 0].imshow(gray_average, cmap='gray')
axes[1, 0].set_title('Anh xam - Phuong phap trung binh\n((R+G+B)/3)',
                      fontsize=14, fontweight='bold')
axes[1, 0].axis('off')

axes[1, 1].imshow(gray_opencv, cmap='gray')
axes[1, 1].set_title('Anh xam - OpenCV\n(cv2.cvtColor)',
                      fontsize=14, fontweight='bold')
axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig('chuyen_doi_anh_xam.png', dpi=150, bbox_inches='tight')
plt.show()

# CHUYEN DOI ANH XAM SANG ANH MAU

def grayscale_to_rgb(gray_img):
    # Tao anh RGB gia lap tu anh xam (R=G=B)
    h, w = gray_img.shape
    rgb_img = np.zeros((h, w, 3), dtype=np.uint8)
    rgb_img[:, :, 0] = gray_img
    rgb_img[:, :, 1] = gray_img
    rgb_img[:, :, 2] = gray_img
    return rgb_img

rgb_from_gray = grayscale_to_rgb(gray_standard)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].imshow(original_image)
axes[0].set_title('Anh RGB goc', fontsize=14, fontweight='bold')
axes[0].axis('off')

axes[1].imshow(gray_standard, cmap='gray')
axes[1].set_title('Anh xam', fontsize=14, fontweight='bold')
axes[1].axis('off')

axes[2].imshow(rgb_from_gray)
axes[2].set_title('Anh RGB tu anh xam\n(Pseudo-color: R=G=B=Gray)',
                  fontsize=14, fontweight='bold')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('chuyen_doi_gray_to_rgb.png', dpi=150, bbox_inches='tight')
plt.show()

# TACH VA HIEN THI TUNG KENH MAU

# Tach 3 kenh mau rieng biet
red_channel = original_image[:, :, 0]
green_channel = original_image[:, :, 1]
blue_channel = original_image[:, :, 2]

fig, axes = plt.subplots(2, 4, figsize=(20, 10))

# Hang 1: Cac kenh duoi dang xam
axes[0, 0].imshow(original_image)
axes[0, 0].set_title('Anh RGB goc', fontsize=12, fontweight='bold')
axes[0, 0].axis('off')

axes[0, 1].imshow(red_channel, cmap='gray')
axes[0, 1].set_title('Kenh Red (grayscale)', fontsize=12, fontweight='bold')
axes[0, 1].axis('off')

axes[0, 2].imshow(green_channel, cmap='gray')
axes[0, 2].set_title('Kenh Green (grayscale)', fontsize=12, fontweight='bold')
axes[0, 2].axis('off')

axes[0, 3].imshow(blue_channel, cmap='gray')
axes[0, 3].set_title('Kenh Blue (grayscale)', fontsize=12, fontweight='bold')
axes[0, 3].axis('off')

# Hang 2: Cac kenh voi mau tuong ung
red_only = np.zeros_like(original_image)
red_only[:, :, 0] = red_channel

green_only = np.zeros_like(original_image)
green_only[:, :, 1] = green_channel

blue_only = np.zeros_like(original_image)
blue_only[:, :, 2] = blue_channel

axes[1, 0].imshow(original_image)
axes[1, 0].set_title('Anh RGB goc', fontsize=12, fontweight='bold')
axes[1, 0].axis('off')

axes[1, 1].imshow(red_only)
axes[1, 1].set_title('Chi kenh Red\n(G=0, B=0)', fontsize=12, fontweight='bold')
axes[1, 1].axis('off')

axes[1, 2].imshow(green_only)
axes[1, 2].set_title('Chi kenh Green\n(R=0, B=0)', fontsize=12, fontweight='bold')
axes[1, 2].axis('off')

axes[1, 3].imshow(blue_only)
axes[1, 3].set_title('Chi kenh Blue\n(R=0, G=0)', fontsize=12, fontweight='bold')
axes[1, 3].axis('off')

plt.tight_layout()
plt.savefig('tach_kenh_mau.png', dpi=150, bbox_inches='tight')
plt.show()

# Thong ke tung kenh
print("\nPhan tich tung kenh mau:")
print(f"Kenh Red   - Min: {red_channel.min()}, Max: {red_channel.max()}, Mean: {red_channel.mean():.2f}")
print(f"Kenh Green - Min: {green_channel.min()}, Max: {green_channel.max()}, Mean: {green_channel.mean():.2f}")
print(f"Kenh Blue  - Min: {blue_channel.min()}, Max: {blue_channel.max()}, Mean: {blue_channel.mean():.2f}")

# KET HOP CAC KENH DE TAI TAO ANH

# Ghep lai 3 kenh thanh anh RGB
reconstructed_image = np.stack([red_channel, green_channel, blue_channel], axis=2)
is_identical = np.array_equal(original_image, reconstructed_image)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].imshow(original_image)
axes[0].set_title('Anh goc', fontsize=14, fontweight='bold')
axes[0].axis('off')

axes[1].imshow(reconstructed_image)
axes[1].set_title('Anh tai tao tu 3 kenh\n(R + G + B)', fontsize=14, fontweight='bold')
axes[1].axis('off')

# Tinh sai khac
difference = cv2.absdiff(original_image, reconstructed_image)
axes[2].imshow(difference)
axes[2].set_title('Su khac biet\n(Absolute Difference)', fontsize=14, fontweight='bold')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('tai_tao_anh.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nAnh goc va anh tai tao giong nhau 100%: {is_identical}")
print(f"Tong sai khac: {np.sum(difference)}")

# TAO ANH MOI BANG CACH HOAN DOI KENH MAU

# Tao cac to hop khac nhau cua 3 kenh
bgr_image = np.stack([blue_channel, green_channel, red_channel], axis=2)
brg_image = np.stack([blue_channel, red_channel, green_channel], axis=2)
gbr_image = np.stack([green_channel, blue_channel, red_channel], axis=2)
grb_image = np.stack([green_channel, red_channel, blue_channel], axis=2)
rbg_image = np.stack([red_channel, blue_channel, green_channel], axis=2)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

axes[0, 0].imshow(original_image)
axes[0, 0].set_title('Anh goc (RGB)', fontsize=12, fontweight='bold')
axes[0, 0].axis('off')

axes[0, 1].imshow(bgr_image)
axes[0, 1].set_title('BGR', fontsize=12, fontweight='bold')
axes[0, 1].axis('off')

axes[0, 2].imshow(brg_image)
axes[0, 2].set_title('BRG', fontsize=12, fontweight='bold')
axes[0, 2].axis('off')

axes[1, 0].imshow(gbr_image)
axes[1, 0].set_title('GBR', fontsize=12, fontweight='bold')
axes[1, 0].axis('off')

axes[1, 1].imshow(grb_image)
axes[1, 1].set_title('GRB', fontsize=12, fontweight='bold')
axes[1, 1].axis('off')

axes[1, 2].imshow(rbg_image)
axes[1, 2].set_title('RBG', fontsize=12, fontweight='bold')
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig('hoan_doi_kenh_mau.png', dpi=150, bbox_inches='tight')
plt.show()

# KET HOP NHIEU ANH

if len(images_dict) >= 2:
    # Lay 2 anh dau tien
    img1 = images_dict[image_names[0]]
    img2 = images_dict[image_names[1]]
    
    # Resize anh 2 cho bang anh 1
    img2_resized = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    # Ket hop: R tu anh 1, G+B tu anh 2
    hybrid1 = np.zeros_like(img1)
    hybrid1[:, :, 0] = img1[:, :, 0]
    hybrid1[:, :, 1] = img2_resized[:, :, 1]
    hybrid1[:, :, 2] = img2_resized[:, :, 2]
    
    # Ket hop: R+G tu anh 1, B tu anh 2
    hybrid2 = np.zeros_like(img1)
    hybrid2[:, :, 0] = img1[:, :, 0]
    hybrid2[:, :, 1] = img1[:, :, 1]
    hybrid2[:, :, 2] = img2_resized[:, :, 2]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    
    axes[0, 0].imshow(img1)
    axes[0, 0].set_title(f'Anh 1: {image_names[0]}', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(img2_resized)
    axes[0, 1].set_title(f'Anh 2: {image_names[1]}', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(hybrid1)
    axes[1, 0].set_title('Hybrid 1\nR(Anh1) + G(Anh2) + B(Anh2)',
                         fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(hybrid2)
    axes[1, 1].set_title('Hybrid 2\nR(Anh1) + G(Anh1) + B(Anh2)',
                         fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('ket_hop_nhieu_anh.png', dpi=150, bbox_inches='tight')
    plt.show()

# THAO TAC VOI KENH - TANG/GIAM CUONG DO

def adjust_channel(img, channel_idx, factor):
    # Nhan gia tri kenh voi he so
    result = img.copy().astype(np.float32)
    result[:, :, channel_idx] *= factor
    # Dam bao gia tri trong khoang 0-255
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result

# Tao cac bien the khac nhau
red_boosted = adjust_channel(original_image, 0, 1.5)
green_boosted = adjust_channel(original_image, 1, 1.5)
blue_boosted = adjust_channel(original_image, 2, 1.5)
red_reduced = adjust_channel(original_image, 0, 0.5)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

axes[0, 0].imshow(original_image)
axes[0, 0].set_title('Anh goc', fontsize=12, fontweight='bold')
axes[0, 0].axis('off')

axes[0, 1].imshow(red_boosted)
axes[0, 1].set_title('Tang kenh Red (x1.5)\nAnh am hon',
                     fontsize=12, fontweight='bold')
axes[0, 1].axis('off')

axes[0, 2].imshow(green_boosted)
axes[0, 2].set_title('Tang kenh Green (x1.5)\nAnh xanh hon',
                     fontsize=12, fontweight='bold')
axes[0, 2].axis('off')

axes[1, 0].imshow(blue_boosted)
axes[1, 0].set_title('Tang kenh Blue (x1.5)\nAnh lanh hon',
                     fontsize=12, fontweight='bold')
axes[1, 0].axis('off')

axes[1, 1].imshow(red_reduced)
axes[1, 1].set_title('Giam kenh Red (x0.5)\nAnh xanh lam hon',
                     fontsize=12, fontweight='bold')
axes[1, 1].axis('off')

# Loai bo hoan toan 1 kenh
no_red = original_image.copy()
no_red[:, :, 0] = 0

axes[1, 2].imshow(no_red)
axes[1, 2].set_title('Loai bo kenh Red (R=0)\nChi con Cyan',
                     fontsize=12, fontweight='bold')
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig('dieu_chinh_kenh_mau.png', dpi=150, bbox_inches='tight')
plt.show()

# HISTOGRAM CUA CAC KENH MAU

# Ve histogram cho 3 kenh
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

axes[0, 0].imshow(original_image)
axes[0, 0].set_title('Anh goc', fontsize=12, fontweight='bold')
axes[0, 0].axis('off')

axes[0, 1].hist(red_channel.ravel(), bins=256, range=(0, 256),
                color='red', alpha=0.7)
axes[0, 1].set_title('Histogram - Kenh Red', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Gia tri pixel')
axes[0, 1].set_ylabel('So luong pixel')
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].hist(green_channel.ravel(), bins=256, range=(0, 256),
                color='green', alpha=0.7)
axes[1, 0].set_title('Histogram - Kenh Green', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Gia tri pixel')
axes[1, 0].set_ylabel('So luong pixel')
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].hist(blue_channel.ravel(), bins=256, range=(0, 256),
                color='blue', alpha=0.7)
axes[1, 1].set_title('Histogram - Kenh Blue', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Gia tri pixel')
axes[1, 1].set_ylabel('So luong pixel')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('histogram_cac_kenh.png', dpi=150, bbox_inches='tight')
plt.show()

# TAO HIEU UNG MAU DAC BIET

def sepia_effect(img):
    # Ma tran bien doi Sepia
    sepia_filter = np.array([[0.393, 0.769, 0.189],
                             [0.349, 0.686, 0.168],
                             [0.272, 0.534, 0.131]])
    sepia_img = img.dot(sepia_filter.T)
    sepia_img = np.clip(sepia_img, 0, 255).astype(np.uint8)
    return sepia_img

def negative_effect(img):
    # Dao nguoc gia tri pixel
    return 255 - img

def keep_only_color(img, color='red'):
    # Chi giu 1 mau, cac mau khac chuyen xam
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    result = np.stack([gray, gray, gray], axis=2)
    
    if color == 'red':
        result[:, :, 0] = img[:, :, 0]
    elif color == 'green':
        result[:, :, 1] = img[:, :, 1]
    elif color == 'blue':
        result[:, :, 2] = img[:, :, 2]
    
    return result

# Ap dung cac hieu ung
sepia_img = sepia_effect(original_image)
negative_img = negative_effect(original_image)
red_pop = keep_only_color(original_image, 'red')
green_pop = keep_only_color(original_image, 'green')

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

axes[0, 0].imshow(original_image)
axes[0, 0].set_title('Anh goc', fontsize=12, fontweight='bold')
axes[0, 0].axis('off')

axes[0, 1].imshow(sepia_img)
axes[0, 1].set_title('Hieu ung Sepia\n(Mau nau co dien)',
                     fontsize=12, fontweight='bold')
axes[0, 1].axis('off')

axes[0, 2].imshow(negative_img)
axes[0, 2].set_title('Hieu ung Negative\n(Am ban)',
                     fontsize=12, fontweight='bold')
axes[0, 2].axis('off')

axes[1, 0].imshow(red_pop)
axes[1, 0].set_title('Color Pop - Red\n(Chi giu mau do)',
                     fontsize=12, fontweight='bold')
axes[1, 0].axis('off')

axes[1, 1].imshow(green_pop)
axes[1, 1].set_title('Color Pop - Green\n(Chi giu mau xanh la)',
                     fontsize=12, fontweight='bold')
axes[1, 1].axis('off')

# Tao gradient RGB
h, w = 300, 300
gradient = np.zeros((h, w, 3), dtype=np.uint8)
for i in range(h):
    for j in range(w):
        gradient[i, j, 0] = int(255 * i / h)
        gradient[i, j, 1] = int(255 * j / w)
        gradient[i, j, 2] = int(128)

axes[1, 2].imshow(gradient)
axes[1, 2].set_title('Gradient RGB tu tao\n(R: doc, G: ngang)',
                     fontsize=12, fontweight='bold')
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig('hieu_ung_mau_dac_biet.png', dpi=150, bbox_inches='tight')
plt.show()
