import cv2
import os
import numpy as np

# Create kernels
sharp = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
morph = np.ones((5, 5), np.uint8)

# Load image
original = cv2.imread("baseball.jpg")

# Convert to grayscale and the apply threshold (black and white)
im = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
im = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                           cv2.THRESH_BINARY, 11, 2)

# Invert image
im = cv2.bitwise_not(im)


def dilate(im, morph):
    # Grow main features
    im = cv2.dilate(im, morph)
    return im


def erode(im, morph):
    im = cv2.erode(im, morph)
    return im


def contour(im):
    # Find rectangles in image
    contours, hierarchy = cv2.findContours(im, 1, 2)

    # Select only 9 largest rectangles
    largest_rects = sorted(contours, key=lambda x: cv2.contourArea(x))[-9:]

    return largest_rects


largest_rects = contour(im)
rect_areas = [cv2.contourArea(x) for x in largest_rects]
average_area = sum(rect_areas) / 9


def check_areas(largest_rects, rectangle_areas, im, morph, second=False):
    for rect in rectangle_areas:
        diff = abs(rect - average_area)
        if diff > .25 * average_area:
            if second:
                return None, None
            else:
                second = True
            im = erode(im, morph)
            largest_rects = contour(im)
            largest_rects, im = check_areas(largest_rects, rectangle_areas, im,
                                            morph, second)
            break
    return largest_rects, im


largest_rects, im = check_areas(largest_rects, rect_areas, im, morph)
if im is None:
    raise AssertionError("Rectangles found were not of similar size.")
# Create grouping for top left point of rectangles and the rectangle itself
r_points = []
for rect in largest_rects:
    x, y, *_ = cv2.boundingRect(rect)
    r_points.append((rect, y, x))

# Sort rectangles by y position in image
r_points.sort(key=lambda x: x[1])

# Loop through and sort the three different rows of rectangles
for x in range(0, len(r_points), 3):
    temp = r_points[x:x + 3]
    temp.sort(key=lambda r: r[2])
    r_points[x:x + 3] = temp

largest_rects = [rect[0] for rect in r_points]
r_points = [rect[1:] for rect in r_points]

# Make sure the squares directory exists
if not os.path.exists('squares/'):
    os.mkdir('squares')

# Resize selected rectangle images and save them to squares
for i, cnt in enumerate(largest_rects, 1):
    x, y, w, h = cv2.boundingRect(cnt)
    small = original[y:y + h, x:x + w]
    # small = cv2.resize(small, (100, 100))
    cv2.imwrite('squares/square_' + str(i) + '.png', small)

# Draw rectangles on the original image
for i, cnt in enumerate(largest_rects[-9:], 1):
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(im, (x, y), (x + w, y + h), 100, 2)

# Write the original image with drawn rectangles to a new image
cv2.imwrite('processed.png', im)
