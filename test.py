import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage import filters, morphology, measure
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage as ndi


# ===============================
# 1. Petri dish (circle) detection
# ===============================
def detect_petri_dishes(image):
    """
    Detect circular petri dishes using Hough Circle Transform
    """
    scale = 0.4
    small = cv2.resize(
        image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
    )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 1.5)
    
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=200 * scale,
        param1=100,
        param2=60,
        minRadius=int(1000*scale),
        maxRadius=int(1500*scale)
    )

    if circles is None:
        return []

    circles = np.round(circles[0]).astype(int)
    results = []
    for x, y, r in circles:
        results.append((
            int(x / scale),
            int(y / scale),
            int(r / scale)
        ))

    return circles


# ===============================
# 2. Colony counting in one plate
# ===============================
def count_colonies(plate_img):
    """
    Count bacterial colonies inside a single petri dish
    """
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

    # contrast enhancement
    gray = cv2.equalizeHist(gray)

    # adaptive threshold (Otsu)
    thresh = filters.threshold_otsu(gray)
    binary = gray < thresh   # colonies usually darker

    # remove noise
    binary = morphology.remove_small_objects(binary, max_size=50)
    binary = morphology.opening(binary, morphology.disk(2))

    # distance transform
    distance = ndi.distance_transform_edt(binary)

    # local maxima
    coords = peak_local_max(
        distance,
        min_distance=10,
        labels=binary
    )

    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True

    markers, _ = ndi.label(mask)

    # watershed segmentation
    labels = watershed(-distance, markers, mask=binary)

    colony_count = len(np.unique(labels)) - 1
    return colony_count, labels


# ===============================
# 3. Main pipeline
# ===============================
def process_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found")

    output = image.copy()
    print("before detect_petri_dishes\n")
    plates = detect_petri_dishes(image)
    print("after detect_petri_dishes\n")

    results = []

    for i, (x, y, r) in enumerate(plates):
        # crop plate
        y1, y2 = max(0, y - r), min(image.shape[0], y + r)
        x1, x2 = max(0, x - r), min(image.shape[1], x + r)
        plate_img = image[y1:y2, x1:x2]

        print("before count_colonies\n")
        count, labels = count_colonies(plate_img)
        print("after count_colonies\n")
        results.append(count)

        # draw result
        cv2.circle(output, (x, y), r, (0, 255, 0), 3)
        cv2.putText(
            output,
            f"Plate {i+1}: {count}",
            (x - r, y - r - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    return output, results


# ===============================
# 4. Run
# ===============================
if __name__ == "__main__":
    image_path = "ref.jpg"   
    print("before process_image\n")
    output, counts = process_image(image_path)

    print("Colony counts per plate:")
    for i, c in enumerate(counts):
        print(f"Plate {i+1}: {c}")

    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.savefig("result.png", dpi=200)
    print("Saved the result")
