import cv2
import os
import sys
import numpy as np


def draw_contours(im, contours, file_name=None):
    # Draw rectangles and write an image
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(im, (x, y), (x + w, y + h), 100, 1)
    file_name = file_name or 'processed_image.png'
    cv2.imwrite(file_name, im)


def dilate(im, kernel):
    # Grow main features
    im = cv2.dilate(im, kernel)
    return im


def erode(im, kernel):
    im = cv2.erode(im, kernel)
    return im


def contours(im):
    # Find rectangles in image
    contours, _ = cv2.findContours(im, 1, 2)

    # Select only 9 largest rectangles
    largest_contours = sorted(contours, key=lambda x: cv2.contourArea(x))[-9:]
    return largest_contours


def print_areas():
    for t in ("Eroded:", "Dilated:", "Regular:"):
        yield t


area_names = print_areas()


def contour_area(contour):
    vals = []
    for point in contour:
        vals.append((point[0][0], point[0][1]))
    x_min = min(vals, key=lambda x: x[0])[0]
    x_max = max(vals, key=lambda x: x[0])[0]
    y_min = min(vals, key=lambda x: x[1])[1]
    y_max = max(vals, key=lambda x: x[1])[1]

    print(x_min, y_min, x_max, y_max)
    area = (x_max - x_min) * (y_max - y_min)
    return area


def contours_similar(largest_contours):

    contour_areas = [contour_area(x) for x in largest_contours]
    average_area = sum(contour_areas) / 9
    print(next(area_names))
    print(contour_areas, average_area)
    for rect in contour_areas:
        diff = abs(rect - average_area)
        if diff > .25 * average_area:
            return
    return True


def get_ordered_points(largest_contours, get_prime_corner: bool = False):
    # Create grouping for top left point of rectangles and the rectangle itself
    points = []
    for rect in largest_contours:
        x, y, *_ = cv2.boundingRect(rect)
        points.append((rect, y, x))

    # Sort rectangles by y position in image
    points.sort(key=lambda x: x[1])

    # Loop through and sort the three different rows of rectangles
    for x in range(0, len(points), 3):
        temp = points[x:x + 3]
        temp.sort(key=lambda r: r[2])
        points[x:x + 3] = temp

    if get_prime_corner:
        return [(rect[0], rect[1:]) for rect in points]

    return [rect[0] for rect in points]


def save_rectangles(im, largest_contours):
    # Make sure the squares directory exists
    if not os.path.exists('squares/'):
        os.mkdir('squares')

    # Save each individual rectangle to file
    for i, contour in enumerate(largest_contours, 1):
        x, y, w, h = cv2.boundingRect(contour)
        rect = im[y:y + h, x:x + w]
        # rect = cv2.resize(rect, (100, 100))
        cv2.imwrite('squares/square_' + str(i) + '.png', rect)


def best_contours(im, kernel):
    largest_contours = contours(im)
    # If the contours of the current image are not similar in size
    if not contours_similar(largest_contours):
        # erode the image
        eroded_im = erode(im, kernel)
        eroded_contours = contours(eroded_im)
        # If the contours are still not similar
        if not contours_similar(eroded_contours):
            draw_contours(eroded_im, eroded_contours, '1_eroded_image.png')
            # dilate instead
            dilated_im = dilate(im, kernel)
            dilated_contours = contours(dilated_im)
            # If they are still not similar, raise error
            if not contours_similar(dilated_contours):
                draw_contours(dilated_im, dilated_contours,
                              '2_dilated_image.png')
                draw_contours(im, largest_contours, '3_binary_image.png')
                raise AssertionError(
                    "Unsuccessful in finding rectagles of similar size.")

            largest_contours = dilated_contours
        else:
            largest_contours = eroded_contours

    return largest_contours


def get_squares(file_name: str, kernel: np.array = None):
    # Create kernels
    kernel = kernel or np.ones((5, 5), np.uint8)

    # Load image
    original = cv2.imread(file_name)

    # Convert to grayscale and the apply threshold (black and white)
    im = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    im = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY, 11, 2)

    # Invert image
    im = cv2.bitwise_not(im)

    largest_contours = best_contours(im, kernel)
    largest_contours = get_ordered_points(largest_contours)

    save_rectangles(original, largest_contours)

    # Write the original image with drawn rectangles to a new image
    draw_contours(original, largest_contours)


def main():
    args = sys.argv[1:]
    file_name = 'badges.jpeg'
    if args:
        file_name = args[0]

    get_squares(file_name)


if __name__ == "__main__":
    main()
