import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files
from PIL import Image
import io

plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 10

print("✓ Đã import thành công các thư viện cần thiết")

def upload_images():
    """
    Hàm upload ảnh từ máy tính lên Colab
    Returns: dictionary chứa tên file và dữ liệu ảnh
    """
    print("📁 Vui lòng chọn ảnh từ máy tính của bạn...")
    uploaded = files.upload()

    images = {}
    for filename in uploaded.keys():
        # Đọc ảnh từ bytes
        image_data = uploaded[filename]
        img = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        # Chuyển từ BGR sang RGB (OpenCV đọc ảnh theo định dạng BGR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images[filename] = img_rgb
        print(f"✓ Đã tải ảnh: {filename} - Kích thước: {img_rgb.shape}")

    return images

# Upload ảnh (bạn đã có sẵn 3 ảnh)
images_dict = upload_images()

# Lấy ảnh đầu tiên để thực hiện các thao tác
image_names = list(images_dict.keys())
original_image = images_dict[image_names[0]]

print(f"\n📸 Đang làm việc với ảnh: {image_names[0]}")
print(f"   Kích thước: {original_image.shape[0]}x{original_image.shape[1]} pixels")
print(f"   Số kênh màu: {original_image.shape[2]}")

# ====================================================================
# PHẦN 4: CHUYỂN ĐỔI ẢNH MÀU SANG ẢNH XÁM
# ====================================================================

def rgb_to_grayscale_manual(img):
    """
    Chuyển đổi RGB sang Grayscale sử dụng công thức chuẩn
    Gray = 0.299*R + 0.587*G + 0.114*B
    """
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]

    gray = 0.299 * R + 0.587 * G + 0.114 * B
    return gray.astype(np.uint8)

def rgb_to_grayscale_average(img):
    """
    Chuyển đổi RGB sang Grayscale bằng phương pháp trung bình
    Gray = (R + G + B) / 3
    """
    return np.mean(img, axis=2).astype(np.uint8)

# Áp dụng các phương pháp chuyển đổi
gray_standard = rgb_to_grayscale_manual(original_image)
gray_average = rgb_to_grayscale_average(original_image)
gray_opencv = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)

# Hiển thị kết quả
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

axes[0, 0].imshow(original_image)
axes[0, 0].set_title('Ảnh màu gốc (RGB)', fontsize=14, fontweight='bold')
axes[0, 0].axis('off')


axes[0, 1].imshow(gray_standard, cmap='gray')
axes[0, 1].set_title('Ảnh xám - Phương pháp chuẩn\n(0.299R + 0.587G + 0.114B)',
                      fontsize=14, fontweight='bold')
axes[0, 1].axis('off')

axes[1, 0].imshow(gray_average, cmap='gray')
axes[1, 0].set_title('Ảnh xám - Phương pháp trung bình\n((R+G+B)/3)',
                      fontsize=14, fontweight='bold')
axes[1, 0].axis('off')

axes[1, 1].imshow(gray_opencv, cmap='gray')
axes[1, 1].set_title('Ảnh xám - OpenCV\n(cv2.cvtColor)',
                      fontsize=14, fontweight='bold')
axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig('chuyen_doi_anh_xam.png', dpi=150, bbox_inches='tight')
plt.show()


# ====================================================================
# PHẦN 5: CHUYỂN ĐỔI ẢNH XÁM SANG ẢNH MÀU (GIẢ LẬP)
# ====================================================================

def grayscale_to_rgb(gray_img):
    """
    Chuyển ảnh xám thành ảnh RGB (pseudo-color)
    R = G = B = Gray value
    """
    h, w = gray_img.shape
    rgb_img = np.zeros((h, w, 3), dtype=np.uint8)
    rgb_img[:, :, 0] = gray_img  # Red channel
    rgb_img[:, :, 1] = gray_img  # Green channel
    rgb_img[:, :, 2] = gray_img  # Blue channel
    return rgb_img

# Chuyển đổi ngược
rgb_from_gray = grayscale_to_rgb(gray_standard)

# Hiển thị
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].imshow(original_image)
axes[0].set_title('Ảnh RGB gốc', fontsize=14, fontweight='bold')
axes[0].axis('off')

axes[1].imshow(gray_standard, cmap='gray')
axes[1].set_title('Ảnh xám', fontsize=14, fontweight='bold')
axes[1].axis('off')

axes[2].imshow(rgb_from_gray)
axes[2].set_title('Ảnh RGB từ ảnh xám\n(Pseudo-color: R=G=B=Gray)',
                  fontsize=14, fontweight='bold')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('chuyen_doi_gray_to_rgb.png', dpi=150, bbox_inches='tight')
plt.show()

# ====================================================================
# PHẦN 6: TÁCH VÀ HIỂN THỊ TỪNG KÊNH MÀU
# ====================================================================

# Tách các kênh màu
red_channel = original_image[:, :, 0]
green_channel = original_image[:, :, 1]
blue_channel = original_image[:, :, 2]

# Hiển thị từng kênh dưới dạng ảnh xám
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

# Hàng 1: Ảnh gốc và các kênh dưới dạng xám
axes[0, 0].imshow(original_image)
axes[0, 0].set_title('Ảnh RGB gốc', fontsize=12, fontweight='bold')
axes[0, 0].axis('off')

axes[0, 1].imshow(red_channel, cmap='gray')
axes[0, 1].set_title('Kênh Red (grayscale)', fontsize=12, fontweight='bold')
axes[0, 1].axis('off')

axes[0, 2].imshow(green_channel, cmap='gray')
axes[0, 2].set_title('Kênh Green (grayscale)', fontsize=12, fontweight='bold')
axes[0, 2].axis('off')

axes[0, 3].imshow(blue_channel, cmap='gray')
axes[0, 3].set_title('Kênh Blue (grayscale)', fontsize=12, fontweight='bold')
axes[0, 3].axis('off')

# Hàng 2: Các kênh với màu tương ứng
# Tạo ảnh chỉ có kênh Red
red_only = np.zeros_like(original_image)
red_only[:, :, 0] = red_channel

# Tạo ảnh chỉ có kênh Green
green_only = np.zeros_like(original_image)
green_only[:, :, 1] = green_channel

# Tạo ảnh chỉ có kênh Blue
blue_only = np.zeros_like(original_image)
blue_only[:, :, 2] = blue_channel

axes[1, 0].imshow(original_image)
axes[1, 0].set_title('Ảnh RGB gốc', fontsize=12, fontweight='bold')
axes[1, 0].axis('off')

axes[1, 1].imshow(red_only)
axes[1, 1].set_title('Chỉ kênh Red\n(G=0, B=0)', fontsize=12, fontweight='bold')
axes[1, 1].axis('off')

axes[1, 2].imshow(green_only)
axes[1, 2].set_title('Chỉ kênh Green\n(R=0, B=0)', fontsize=12, fontweight='bold')
axes[1, 2].axis('off')

axes[1, 3].imshow(blue_only)
axes[1, 3].set_title('Chỉ kênh Blue\n(R=0, G=0)', fontsize=12, fontweight='bold')
axes[1, 3].axis('off')

plt.tight_layout()
plt.savefig('tach_kenh_mau.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n📊 PHÂN TÍCH TỪNG KÊNH MÀU:")
print(f"Kênh Red   - Min: {red_channel.min()}, Max: {red_channel.max()}, Mean: {red_channel.mean():.2f}")
print(f"Kênh Green - Min: {green_channel.min()}, Max: {green_channel.max()}, Mean: {green_channel.mean():.2f}")
print(f"Kênh Blue  - Min: {blue_channel.min()}, Max: {blue_channel.max()}, Mean: {blue_channel.mean():.2f}")

# ====================================================================
# PHẦN 7: KẾT HỢP CÁC KÊNH ĐỂ TÁI TẠO ẢNH
# ====================================================================

# Tái tạo ảnh từ 3 kênh
reconstructed_image = np.stack([red_channel, green_channel, blue_channel], axis=2)

# Kiểm tra sự giống nhau
is_identical = np.array_equal(original_image, reconstructed_image)

# Hiển thị
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].imshow(original_image)
axes[0].set_title('Ảnh gốc', fontsize=14, fontweight='bold')
axes[0].axis('off')

axes[1].imshow(reconstructed_image)
axes[1].set_title('Ảnh tái tạo từ 3 kênh\n(R + G + B)', fontsize=14, fontweight='bold')
axes[1].axis('off')

# Hiển thị sự khác biệt (nếu có)
difference = cv2.absdiff(original_image, reconstructed_image)
axes[2].imshow(difference)
axes[2].set_title('Sự khác biệt\n(Absolute Difference)', fontsize=14, fontweight='bold')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('tai_tao_anh.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n✓ Ảnh gốc và ảnh tái tạo giống nhau 100%: {is_identical}")
print(f"  Tổng sai khác: {np.sum(difference)}")

# ====================================================================
# PHẦN 8: TẠO ẢNH MỚI BẰNG CÁCH HOÁN ĐỔI KÊNH MÀU
# ====================================================================

# Hoán đổi các kênh màu
rgb_image = original_image.copy()  # R-G-B (gốc)
rbg_image = np.stack([red_channel, blue_channel, green_channel], axis=2)  # R-B-G
grb_image = np.stack([green_channel, red_channel, blue_channel], axis=2)  # G-R-B
gbr_image = np.stack([green_channel, blue_channel, red_channel], axis=2)  # G-B-R
brg_image = np.stack([blue_channel, red_channel, green_channel], axis=2)  # B-R-G
bgr_image = np.stack([blue_channel, green_channel, red_channel], axis=2)  # B-G-R

# Hiển thị tất cả các hoán vị
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

axes[0, 0].imshow(rgb_image)
axes[0, 0].set_title('RGB (Gốc)\nRed-Green-Blue', fontsize=12, fontweight='bold')
axes[0, 0].axis('off')

axes[0, 1].imshow(rbg_image)
axes[0, 1].set_title('RBG\nRed-Blue-Green', fontsize=12, fontweight='bold')
axes[0, 1].axis('off')

axes[0, 2].imshow(grb_image)
axes[0, 2].set_title('GRB\nGreen-Red-Blue', fontsize=12, fontweight='bold')
axes[0, 2].axis('off')

axes[1, 0].imshow(gbr_image)
axes[1, 0].set_title('GBR\nGreen-Blue-Red', fontsize=12, fontweight='bold')
axes[1, 0].axis('off')

axes[1, 1].imshow(brg_image)
axes[1, 1].set_title('BRG\nBlue-Red-Green', fontsize=12, fontweight='bold')
axes[1, 1].axis('off')

axes[1, 2].imshow(bgr_image)
axes[1, 2].set_title('BGR\nBlue-Green-Red', fontsize=12, fontweight='bold')
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig('hoan_doi_kenh_mau.png', dpi=150, bbox_inches='tight')
plt.show()

# ====================================================================
# PHẦN 9: KẾT HỢP KÊNH TỪ NHIỀU ẢNH KHÁC NHAU
# ====================================================================

if len(images_dict) >= 2:
    print("\n📸 Kết hợp kênh từ nhiều ảnh khác nhau...")

    # Lấy 2 ảnh đầu tiên
    img1 = images_dict[image_names[0]]
    img2 = images_dict[image_names[1]]

    # Resize ảnh 2 để cùng kích thước với ảnh 1
    img2_resized = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # Tạo ảnh kết hợp: R từ ảnh 1, G và B từ ảnh 2
    hybrid1 = np.zeros_like(img1)
    hybrid1[:, :, 0] = img1[:, :, 0]  # Red từ ảnh 1
    hybrid1[:, :, 1] = img2_resized[:, :, 1]  # Green từ ảnh 2
    hybrid1[:, :, 2] = img2_resized[:, :, 2]  # Blue từ ảnh 2

    # Tạo ảnh kết hợp: R và G từ ảnh 1, B từ ảnh 2
    hybrid2 = np.zeros_like(img1)
    hybrid2[:, :, 0] = img1[:, :, 0]  # Red từ ảnh 1
    hybrid2[:, :, 1] = img1[:, :, 1]  # Green từ ảnh 1
    hybrid2[:, :, 2] = img2_resized[:, :, 2]  # Blue từ ảnh 2

    # Hiển thị
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))

    axes[0, 0].imshow(img1)
    axes[0, 0].set_title(f'Ảnh 1: {image_names[0]}', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(img2_resized)
    axes[0, 1].set_title(f'Ảnh 2: {image_names[1]}', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')

    axes[1, 0].imshow(hybrid1)
    axes[1, 0].set_title('Hybrid 1\nR(Ảnh1) + G(Ảnh2) + B(Ảnh2)',
                         fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(hybrid2)
    axes[1, 1].set_title('Hybrid 2\nR(Ảnh1) + G(Ảnh1) + B(Ảnh2)',
                         fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig('ket_hop_nhieu_anh.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("✓ Đã tạo ảnh kết hợp từ nhiều ảnh nguồn")

# ====================================================================
# PHẦN 10: THAO TÁC VỚI KÊNH - TĂNG/GIẢM CƯỜNG ĐỘ
# ====================================================================

# Tăng/giảm cường độ từng kênh
def adjust_channel(img, channel_idx, factor):
    """
    Điều chỉnh cường độ của một kênh màu
    channel_idx: 0=Red, 1=Green, 2=Blue
    factor: hệ số nhân (>1 để tăng, <1 để giảm)
    """
    result = img.copy().astype(np.float32)
    result[:, :, channel_idx] *= factor
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result

# Tạo các biến thể
red_boosted = adjust_channel(original_image, 0, 1.5)    # Tăng Red
green_boosted = adjust_channel(original_image, 1, 1.5)  # Tăng Green
blue_boosted = adjust_channel(original_image, 2, 1.5)   # Tăng Blue
red_reduced = adjust_channel(original_image, 0, 0.5)    # Giảm Red

# Hiển thị
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

axes[0, 0].imshow(original_image)
axes[0, 0].set_title('Ảnh gốc', fontsize=12, fontweight='bold')
axes[0, 0].axis('off')

axes[0, 1].imshow(red_boosted)
axes[0, 1].set_title('Tăng kênh Red (×1.5)\nẢnh ấm hơn',
                     fontsize=12, fontweight='bold')
axes[0, 1].axis('off')

axes[0, 2].imshow(green_boosted)
axes[0, 2].set_title('Tăng kênh Green (×1.5)\nẢnh xanh hơn',
                     fontsize=12, fontweight='bold')
axes[0, 2].axis('off')

axes[1, 0].imshow(blue_boosted)
axes[1, 0].set_title('Tăng kênh Blue (×1.5)\nẢnh lạnh hơn',
                     fontsize=12, fontweight='bold')
axes[1, 0].axis('off')

axes[1, 1].imshow(red_reduced)
axes[1, 1].set_title('Giảm kênh Red (×0.5)\nẢnh xanh lam hơn',
                     fontsize=12, fontweight='bold')
axes[1, 1].axis('off')

# Loại bỏ hoàn toàn một kênh
no_red = original_image.copy()
no_red[:, :, 0] = 0

axes[1, 2].imshow(no_red)
axes[1, 2].set_title('Loại bỏ kênh Red (R=0)\nChỉ còn Cyan',
                     fontsize=12, fontweight='bold')
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig('dieu_chinh_kenh_mau.png', dpi=150, bbox_inches='tight')
plt.show()

# ====================================================================
# PHẦN 11: HISTOGRAM CỦA CÁC KÊNH MÀU
# ====================================================================

# Tính histogram cho từng kênh
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Hiển thị ảnh gốc
axes[0, 0].imshow(original_image)
axes[0, 0].set_title('Ảnh gốc', fontsize=12, fontweight='bold')
axes[0, 0].axis('off')

# Histogram kênh Red
axes[0, 1].hist(red_channel.ravel(), bins=256, range=(0, 256),
                color='red', alpha=0.7)
axes[0, 1].set_title('Histogram - Kênh Red', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Giá trị pixel')
axes[0, 1].set_ylabel('Số lượng pixel')
axes[0, 1].grid(True, alpha=0.3)

# Histogram kênh Green
axes[1, 0].hist(green_channel.ravel(), bins=256, range=(0, 256),
                color='green', alpha=0.7)
axes[1, 0].set_title('Histogram - Kênh Green', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Giá trị pixel')
axes[1, 0].set_ylabel('Số lượng pixel')
axes[1, 0].grid(True, alpha=0.3)

# Histogram kênh Blue
axes[1, 1].hist(blue_channel.ravel(), bins=256, range=(0, 256),
                color='blue', alpha=0.7)
axes[1, 1].set_title('Histogram - Kênh Blue', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Giá trị pixel')
axes[1, 1].set_ylabel('Số lượng pixel')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('histogram_cac_kenh.png', dpi=150, bbox_inches='tight')
plt.show()
# ====================================================================
# PHẦN 12: TẠO HIỆU ỨNG MÀU ĐẶC BIỆT
# ====================================================================

# Hiệu ứng Sepia (màu nâu cổ điển)
def sepia_effect(img):
    """
    Tạo hiệu ứng Sepia cho ảnh
    """
    sepia_filter = np.array([[0.393, 0.769, 0.189],
                             [0.349, 0.686, 0.168],
                             [0.272, 0.534, 0.131]])

    sepia_img = img.dot(sepia_filter.T)
    sepia_img = np.clip(sepia_img, 0, 255).astype(np.uint8)
    return sepia_img

# Hiệu ứng Negative (âm bản)
def negative_effect(img):
    """
    Tạo hiệu ứng âm bản
    """
    return 255 - img

# Hiệu ứng chỉ giữ 1 màu
def keep_only_color(img, color='red'):
    """
    Chỉ giữ lại một màu, các màu khác chuyển sang xám
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    result = np.stack([gray, gray, gray], axis=2)

    if color == 'red':
        result[:, :, 0] = img[:, :, 0]
    elif color == 'green':
        result[:, :, 1] = img[:, :, 1]
    elif color == 'blue':
        result[:, :, 2] = img[:, :, 2]

    return result

# Áp dụng các hiệu ứng
sepia_img = sepia_effect(original_image)
negative_img = negative_effect(original_image)
red_pop = keep_only_color(original_image, 'red')
green_pop = keep_only_color(original_image, 'green')

# Hiển thị
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

axes[0, 0].imshow(original_image)
axes[0, 0].set_title('Ảnh gốc', fontsize=12, fontweight='bold')
axes[0, 0].axis('off')

axes[0, 1].imshow(sepia_img)
axes[0, 1].set_title('Hiệu ứng Sepia\n(Màu nâu cổ điển)',
                     fontsize=12, fontweight='bold')
axes[0, 1].axis('off')

axes[0, 2].imshow(negative_img)
axes[0, 2].set_title('Hiệu ứng Negative\n(Âm bản)',
                     fontsize=12, fontweight='bold')
axes[0, 2].axis('off')

axes[1, 0].imshow(red_pop)
axes[1, 0].set_title('Color Pop - Red\n(Chỉ giữ màu đỏ)',
                     fontsize=12, fontweight='bold')
axes[1, 0].axis('off')

axes[1, 1].imshow(green_pop)
axes[1, 1].set_title('Color Pop - Green\n(Chỉ giữ màu xanh lá)',
                     fontsize=12, fontweight='bold')
axes[1, 1].axis('off')

# Tạo gradient màu
h, w = 300, 300
gradient = np.zeros((h, w, 3), dtype=np.uint8)
for i in range(h):
    for j in range(w):
        gradient[i, j, 0] = int(255 * i / h)      # Red gradient
        gradient[i, j, 1] = int(255 * j / w)      # Green gradient
        gradient[i, j, 2] = int(128)               # Blue constant

axes[1, 2].imshow(gradient)
axes[1, 2].set_title('Gradient RGB tự tạo\n(R: dọc, G: ngang)',
                     fontsize=12, fontweight='bold')
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig('hieu_ung_mau_dac_biet.png', dpi=150, bbox_inches='tight')
plt.show()

if len(images_dict) >= 2:
    print("   ✓ ket_hop_nhieu_anh.png")

print("\n✅ HOÀN THÀNH BÀI TẬP!")
print("="*70)