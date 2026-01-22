import cv2
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. 이미지 로드
# -----------------------------
img = cv2.imread("ref.jpg")
orig = img.copy()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# -----------------------------
# 2. 페트리 접시 검출 (Hough Circle)
# -----------------------------
gray_blur = cv2.GaussianBlur(gray, (9, 9), 1.5)

circles = cv2.HoughCircles(
    gray_blur,
    cv2.HOUGH_GRADIENT,
    dp=1.2,
    minDist=500,
    param1=100,
    param2=30,
    minRadius=300,
    maxRadius=600
)

dish_mask = np.zeros_like(gray)

if circles is not None:
    circles = np.uint16(np.around(circles))
    x, y, r = circles[0][0]
    cv2.circle(dish_mask, (x, y), r, 255, -1)  # 접시 내부만 흰색

# -----------------------------
# 3. 접시 내부에서 colony 분리
# -----------------------------
# colony는 배경보다 밝기 차이가 있음
# adaptive threshold가 훨씬 안정적
th = cv2.adaptiveThreshold(
    gray,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    51,
    -5
)

# 접시 영역만 사용
colony_mask = cv2.bitwise_and(th, th, mask=dish_mask)

# -----------------------------
# 4. Morphology (colony 정제)
# -----------------------------
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# 노이즈 제거
colony_mask = cv2.morphologyEx(colony_mask, cv2.MORPH_OPEN, kernel)

# colony 내부 채우기 (여기서 fill!)
colony_mask = cv2.morphologyEx(colony_mask, cv2.MORPH_CLOSE, kernel)

# -----------------------------
# 5. Connected Components
# -----------------------------
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
    colony_mask, connectivity=8
)

count = 0
output = orig.copy()

for i in range(1, num_labels):  # 0은 background
    area = stats[i, cv2.CC_STAT_AREA]

    if 30 < area < 3000:  # colony 크기 필터 (중요)
        count += 1
        cx, cy = centroids[i]
        cv2.circle(output, (int(cx), int(cy)), 5, (0, 0, 255), -1)

# -----------------------------
# 6. 결과 시각화
# -----------------------------
plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.title("Original")
plt.imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(1,3,2)
plt.title("Colony Mask")
plt.imshow(colony_mask, cmap="gray")
plt.axis("off")

plt.subplot(1,3,3)
plt.title(f"Detected Colonies: {count}")
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.show()
