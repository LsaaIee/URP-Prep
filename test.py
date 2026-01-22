import cv2
import numpy as np
import matplotlib.pyplot as plt

# =============================
# 1. 이미지 로드
# =============================
img = cv2.imread("single_petri.jpg")
if img is None:
    raise RuntimeError("이미지를 불러올 수 없습니다.")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
h, w = gray.shape

# =============================
# 2. 페트리 접시 마스크 생성
# (이미지 중앙의 가장 큰 원)
# =============================
mask = np.zeros_like(gray, dtype=np.uint8)

center_x, center_y = w // 2, h // 2
radius = min(center_x, center_y) - 20   # 테두리 제거 여유

cv2.circle(mask, (center_x, center_y), radius, 255, -1)

plate = cv2.bitwise_and(gray, gray, mask=mask)

# =============================
# 3. Colony 강조
# =============================
blur = cv2.GaussianBlur(plate, (5, 5), 0)

thresh = cv2.adaptiveThreshold(
    blur,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    31,
    4
)

# =============================
# 4. 노이즈 제거
# =============================
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

# =============================
# 5. Colony contour 검출
# =============================
contours, _ = cv2.findContours(
    thresh,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

colony_count = 0
output = img.copy()

for c in contours:
    area = cv2.contourArea(c)

    # ⚠️ colony 크기 필터 (중요)
    if 30 < area < 3000:
        colony_count += 1
        cv2.drawContours(output, [c], -1, (0, 0, 255), 1)

# =============================
# 6. 결과 출력
# =============================
print(f"Colony 개수: {colony_count}")

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title("Original")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Threshold")
plt.imshow(thresh, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title(f"Detected Colonies: {colony_count}")
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.tight_layout()
plt.show()
