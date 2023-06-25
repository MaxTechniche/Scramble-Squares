import time
import cv2
import os
import shutil
import sys
import numpy as np
import glob
import matplotlib.pyplot as plt
from statistics import mean
from pprint import pprint
from itertools import product



# TODO - Make a function that pairs groups of contours together
# TODO - Improve template matching function


def draw_contours(im, contours, file_name=None):
    # Draw rectangles and write an image
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(im, (x, y), (x + w, y + h), 100, 2)
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
    image_size = im.shape
    image_area = image_size[0] * image_size[1]
    contours, _ = cv2.findContours(im, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = list(contours)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x))[-100:]
    
    while True:
        for i, contour in enumerate(contours):
            if cv2.contourArea(contour) > image_area / 9:
                contours.pop(i)
                break
            elif cv2.contourArea(contour) < image_area / 100:
                contours.pop(i)
                break
        else:
            break
    
    while True:
        brake = False
        for i, contour1 in enumerate(contours):
            x1, y1, w1, h1 = cv2.boundingRect(contour1)
            center1 = (x1 + w1 / 2, y1 + h1 / 2)
            for j, contour2 in enumerate(contours):
                if i == j:
                    continue
                # Get the center of the contour
                x2, y2, w2, h2 = cv2.boundingRect(contour2)
                center2 = (x2 + w2 / 2, y2 + h2 / 2)
                if center1[0] > x2 and center1[0] < x2 + w2 and center1[1] > y2 and center1[1] < y2 + h2:
                    contours.pop(i)
                    brake = True
                elif center2[0] > x1 and center2[0] < x1 + w1 and center2[1] > y1 and center2[1] < y1 + h1:
                    contours.pop(j)
                    brake = True
                if brake:
                    break
            if brake:
                break
                
        else:
            break

    if len(contours) < 9:
        raise AssertionError("Program did not find 9 contours.")

    return contours[-9:]


def print_areas():
    for t in (
            "Regular:",
            "Eroded:",
            "Dilated:",
    ):
        yield t


area_names = print_areas()


def contour_area(contour):
    x, y, w, h = cv2.boundingRect(contour)
    return w * h


def contours_similar(largest_contours):
    contour_areas = [contour_area(x) for x in largest_contours]
    average_area = sum(contour_areas) / 9
    for rect in contour_areas:
        diff = abs(rect - average_area)
        if diff > .10 * average_area:
            return
    return True


def get_ordered_points(largest_contours, get_prime_corner: bool = False):
    # Create grouping for top left point of rectangles and the rectangle itself
    points = []
    for rect in largest_contours:
        x, y, *_ = cv2.boundingRect(rect)
        points.append((rect, y, x))

    # Sort rectangles by y position in image
    points.sort(key=lambda r: r[1])

    # Loop through and sort the three different rows of rectangles
    for x in range(0, len(points), 3):
        temp = points[x:x + 3]
        temp.sort(key=lambda r: r[2])
        points[x:x + 3] = temp

    if get_prime_corner:
        return [(rect[0], rect[1:]) for rect in points]

    return [rect[0] for rect in points]


def save_rectangles(im, largest_contours, img_resize=None):
    # Make sure the squares directory exists
    if not os.path.exists('squares/'):
        os.mkdir('squares')

    # Save each individual rectangle to file
    for i, contour in enumerate(largest_contours, 1):
        x, y, w, h = cv2.boundingRect(contour)
        rect = im[y:y + h, x:x + w]
        if img_resize is not None:
            rect = cv2.resize(rect, (img_resize, img_resize))
        cv2.imwrite('squares/square_' + str(i) + '.png', rect)


def best_contours(im, kernel):
    # Remove previous
    if os.path.exists('squares'):
        shutil.rmtree('squares')
    if os.path.exists('1_binary_image.png'):
        os.remove('1_binary_image.png')
    if os.path.exists('2_eroded_image.png'):
        os.remove('2_eroded_image.png')
    if os.path.exists('3_dilated_image.png'):
        os.remove('3_dilated_image.png')
    if os.path.exists('processed_image.png'):
        os.remove('processed_image.png')

    largest_contours = contours(im)
    draw_contours(im, largest_contours, '1_binary_image.png')
    # If the contours of the current image are not similar in size
    if not contours_similar(largest_contours):
        # erode the image
        eroded_im = erode(im, kernel)
        eroded_contours = contours(eroded_im)
        draw_contours(eroded_im, eroded_contours, '2_eroded_image.png')
        # If the contours are still not similar
        if not contours_similar(eroded_contours):
            # dilate instead
            dilated_im = dilate(im, kernel)
            dilated_contours = contours(dilated_im)
            draw_contours(dilated_im, dilated_contours, '3_dilated_image.png')
            # If they are still not similar, raise error
            if not contours_similar(dilated_contours):
                raise AssertionError(
                    "Unsuccessful in finding rectagles of similar size.")

            largest_contours = dilated_contours
        else:
            largest_contours = eroded_contours

    return largest_contours


def save_slices(im, s_num, slice_width=.15, buffer=.05, length_buffer=.40):
    
    # im = cv2.GaussianBlur(im, (5, 5), 0)

    im_size = im.shape
    slice_width = int(im_size[0] * slice_width)
    buffer = int(im_size[0] * buffer)
    length_buffer = int(im_size[0] * length_buffer)

    top = im[buffer:(slice_width + buffer), length_buffer:-length_buffer]
    cv2.imwrite(f'squares/square_{s_num}_a.png', top)
    right = im[length_buffer:-length_buffer, -(slice_width + buffer):-buffer]
    right = np.rot90(right)
    cv2.imwrite(f'squares/square_{s_num}_b.png', right)
    bottom = im[-(slice_width + buffer):-buffer, length_buffer:-length_buffer]
    bottom = np.rot90(bottom, k=2)
    cv2.imwrite(f'squares/square_{s_num}_c.png', bottom)
    left = im[length_buffer:-length_buffer, buffer:(slice_width + buffer)]
    left = np.rot90(left, k=3)
    cv2.imwrite(f'squares/square_{s_num}_d.png', left)


def squares_exist():
    for s in range(1, 10):
        if not os.path.exists(f"squares/square_{s}.png"):
            raise FileNotFoundError(f"Square {s} was not found.")
    return True


def split_squares():
    if not squares_exist():
        raise AssertionError("""How did you receive this error?  \
            Squares were not found, but you've chosen to go on?""")

    for square in os.listdir('squares'):
        s_num = square[-5]
        im = cv2.imread(f'squares/{square}')
        save_slices(im, s_num)


def get_squares(file_name: str, bin_thresh: int = 255, thresh_block_size: int = 99, kernel: np.array = None, blur_size: int = 11, img_resize: int = 100):
    # Create kernels
    if kernel is None:
        kernel = np.ones((5, 5), np.uint8)

    # Load image
    original = cv2.imread(file_name)

    # Convert to grayscale and the apply threshold (black and white)
    im = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    im = cv2.GaussianBlur(im, (blur_size, blur_size), 0)
    cv2.imwrite('blurred.png', im)
    canny = cv2.Canny(im, 20, 50)
    cv2.imwrite('canny.png', canny)
    im = cv2.adaptiveThreshold(im, bin_thresh, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, thresh_block_size, 3)
    if bin_thresh != 255:
        im = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, thresh_block_size, 3)

    # im = canny
    
    # Invert image
    im = cv2.bitwise_not(im)

    print(im.shape)

    largest_contours = best_contours(im, kernel)
    if len(largest_contours) < 9:
        raise AssertionError("Program did not find 9 contours.")
    largest_contours = get_ordered_points(largest_contours)

    save_rectangles(original, largest_contours, img_resize)

    # Write the original image with drawn rectangles to a new image
    draw_contours(original, largest_contours)

    split_squares()


def match_templates(file_name, det_thresh=0.8, files=False):
    tile_side_rotations = {}
    matches = []
    for num, letter in product(range(1, 10), ('a', 'b', 'c', 'd')):
        if os.path.exists(f'{num}_{letter}'):
            shutil.rmtree(f'{num}_{letter}')
    for f in glob.glob("match_*"):
        os.remove(f)
    for num, letter in product(range(1, 10), ('a', 'b', 'c', 'd')):
        if files:
            matches_so_far = len(glob.glob("match_*"))
        else:
            matches_so_far = len(matches)
        if matches_so_far < 1 and num > 2:
            raise AssertionError("Matches not being found. Exiting.")
        if num == 2 and letter == 'a' and matches_so_far > 24:
            raise AssertionError("Too many matches being found. Exiting.")
        
        template = cv2.imread(f'squares/square_{num}_{letter}.png', cv2.IMREAD_GRAYSCALE)
        w, h = template.shape[::-1]
        
        for num_2 in range(1, 10):
            if num == num_2:
                continue
            for rot, letter_2 in zip(range(1, 5), ('b', 'c', 'd', 'a')):
                
                if (num_2, letter_2, rot) in tile_side_rotations:
                    img_rgb = tile_side_rotations[(num_2, letter_2, rot)]
                else:
                
                    img_rgb = cv2.imread(f'squares/square_{num_2}.png', cv2.IMREAD_GRAYSCALE)
                    for _ in range(rot):
                        img_rgb = cv2.rotate(img_rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    # img_rgb = cv2.GaussianBlur(img_rgb, (5, 5), 0)
                    
                    half = int(img_rgb.shape[0] / 2)
                    third = int(img_rgb.shape[0] / 3) 
                    quarter = int(img_rgb.shape[0] / 4)
                    img_rgb = img_rgb[:half, :]
                    
                    tile_side_rotations[(num_2, letter_2, rot)] = img_rgb
                    
                res = cv2.matchTemplate(img_rgb, template, cv2.TM_CCOEFF_NORMED)
                
                loc = np.where(res >= det_thresh)
                
                # if len(loc[0]) > 0:
                #     if num < num_2:
                #         lower = f'{num}_{letter}'
                #         higher = f'{num_2}_{letter_2}'
                #     else:
                #         lower = f'{num_2}_{letter_2}'
                #         higher = f'{num}_{letter}'
                    
                for pt in zip(*loc[::-1]):
                    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
                    break
            
                if len(loc[0]) > 0:
                    if files:
                        if not os.path.exists(f'match_{num}_{letter}_{num_2}_{letter_2}.png'):
                            cv2.imwrite(f'match_{num}_{letter}_{num_2}_{letter_2}.png', img_rgb)

                        # if not os.path.exists(f'match_{lower}_{higher}.png'):
                        #     cv2.imwrite(f'match_{lower}_{higher}.png', img_rgb)
                    else:
                        matches.append((num, letter, num_2, letter_2))    
                
                
    # cv2.imwrite('template.png', template)
        
    return matches, tile_side_rotations
    

def group_matches(matches: list = None):
    groups = []
    
    if not matches:
        matches = []
        for f in os.listdir():
            if f.startswith('match_'):
                n1, l1, n2, l2 = f[6:-4].split('_')
                matches.append((n1, l1, n2, l2))
                
    for n1, l1, n2, l2 in matches:
        for group in groups:
            if f'{n1}_{l1}' in group or f'{n2}_{l2}' in group:
                group.add(f'{n1}_{l1}')
                group.add(f'{n2}_{l2}')
                break
        else:
            groups.append({f'{n1}_{l1}', f'{n2}_{l2}'})
    
    while True:
        for group in groups:
            for group_2 in groups:
                if group == group_2:
                    continue
                if len(group.intersection(group_2)) > 0:
                    group.update(group_2)
                    groups.remove(group_2)
                    break
            else:
                continue
            break
        else:
            break
    
    return sum(len(group) for group in groups), groups
    
    
def get_tile_side_rotations(groups):
    
    codes = {
        'a': 4,
        'b': 1,
        'c': 2,
        'd': 3
    }
    
    squares_exist()
    
    tile_side_rotations = {}
    colors = {
        "red": {},
        "green": {},
        "blue": {},
        "average": {},
        "group": {}
    }
    for i, group in enumerate(groups):
        for tile in group:
            num, letter = tile.split('_')
            rot = codes[letter]
            tile_side = cv2.imread(f'squares/square_{num}.png')
            for _ in range(rot):
                tile_side = cv2.rotate(tile_side, cv2.ROTATE_90_COUNTERCLOCKWISE)
            tile_side = tile_side[5:tile_side.shape[0] // 4, tile_side.shape[1] // 3:-tile_side.shape[1] // 3]
            # tile_side[:,:,0] = 0
            # # tile_side[:,:,1] = 0
            # tile_side[:,:,2] = 0
            # cv2.imwrite(f'square_{num}_{letter}.png', tile_side)
            
            b, g, r = tile_side[:, :, 0], tile_side[:, :, 1], tile_side[:, :, 2]
            
            colors["red"][f'{num}_{letter}'] = np.average(r)
            colors["green"][f'{num}_{letter}'] = np.average(g)
            colors["blue"][f'{num}_{letter}'] = np.average(b)
            colors["average"][f'{num}_{letter}'] = np.average([np.average(r), np.average(g), np.average(b)])
            colors["group"][f'{num}_{letter}'] = i
            
            
    for color in colors:
        if color != "group":
            colors[color] = {k: v for k, v in sorted(colors[color].items(), key=lambda item: item[1])}
            for tile in colors[color]:
                print(color, tile, colors[color][tile], colors["group"][tile])
    
    for i, color in enumerate(colors, 1):
        if color == "group":
            continue
        plt.subplot(2, 2, i)
        for group in range(8):
            plt.boxplot([colors[color][tile] for tile in colors[color] if colors["group"][tile] == group], positions=[group])
    plt.savefig('boxplot.png')
    
    return tile_side_rotations


def pair_groups(groups, tile_side_rotations: dict = None):
    
    codes = {
        'a': 4,
        'b': 1,
        'c': 2,
        'd': 3
    }
    
    tile_side_rotations = {}
    for num in range(1, 10):
        for rot, letter in zip(range(1, 5), ('b', 'c', 'd', 'a')):
            tile = cv2.imread(f'squares/square_{num}.png')
            for _ in range(rot):
                tile = cv2.rotate(tile, cv2.ROTATE_90_COUNTERCLOCKWISE)
            tile_side_rotations[(int(num), letter, rot)] = tile[:tile.shape[0] // 3, tile.shape[1] // 3:-tile.shape[1] // 3]
        
    group_colors = {}
    
    group_averages = []
    for i, group in enumerate(groups, 1):
        color_averages = []
        for tile in group:
            num, letter = tile.split('_')
            rot = codes[letter]
            color_averages.append(np.average(tile_side_rotations[(int(num), letter, rot)]))
        group_averages.append((i, mean(color_averages)))
    
    group_averages = sorted(group_averages, key=lambda x: x[1])
    pprint(group_averages)
        
    
def print_colors(file_name):
    colors = {
        "red": [],
        "green": [],
        "blue": [],
        "average": [],
    }
    # for i in range(1, 9):
    #     img = cv2.imread(f'squares/square_{i}.png')
    #     b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    #     colors["red"].append(np.average(r))
    #     colors["green"].append(np.average(g))
    #     colors["blue"].append(np.average(b))
    #     colors["average"].append(np.average([np.average(r), np.average(g), np.average(b)]))
        
    # for i, color in enumerate(colors, 1):
    #     print(colors[color])
    #     print(color, np.std(colors[color]))
    
    big_img = cv2.imread(file_name)
    b, g, r = big_img[:, :, 0], big_img[:, :, 1], big_img[:, :, 2]
    print("red", np.average(r))
    print("green", np.average(g))
    print("blue", np.average(b))
    print("average", np.average([np.average(r), np.average(g), np.average(b)]))
    
    low = zip((2, 1, 0), ("r", "g", "b"))
    low = sorted(low, key=lambda x: np.average(big_img[:, :, x[0]]))
    print(low)
    for i, color in low[-2:]:
        big_img[:, :, i] = 0
    cv2.imwrite('low.png', big_img)
            
                                
class MovedFile(Exception):
    pass       

class FoundMatches(Exception):
    message = "Found matches"


def main(file_name, files):
    args = sys.argv[1:]
    if args:
        file_name = args[0]
    print_colors(file_name)
    # file_name = 'low.png'
        
    if os.path.exists('squares'):
        shutil.rmtree('squares')
        
    for f in os.listdir():
        if f.startswith('match_'):
            os.remove(f)
           
    groups_group = []
            
    # bin_thresholds = range(255, 239, -5)
    kernals = [np.ones((i, i), np.uint8) for i in range(1, 10, 2)]
    thresh_block_sizes = range(3, 30, 2)
    
    det_thresholds = range(95, 69, -5)
    
    tile_side_rotations = None
    groups = None

    bin_thresh = 255
    # for bin_thresh in bin_thresholds:
    try:
        for img_resize in (None,):
            
            for thresh_block_size in thresh_block_sizes:
                for kernel in kernals:
                    print(img_resize, thresh_block_size, kernel.shape, end=' ')
                    try:
                        
                        get_squares(file_name, bin_thresh=255, kernel=kernel, img_resize=img_resize)
                        
                        for det_thresh in det_thresholds:
                            try:
                                det_thresh /= 100
                                print(det_thresh)
                                matches, tile_side_rotations = match_templates(file_name, det_thresh=det_thresh, files=files)
                                
                                matches, groups = group_matches(matches=matches)
                                if matches == 36 and len(groups) == 8:
                                    print('Possible good match parameters found.')
                                    print('bin_thresh:', bin_thresh)
                                    print('thresh_block_size:', thresh_block_size) 
                                    print('kernel.shape:', kernel.shape)
                                    print('det_thresh:', det_thresh)
                                    print('Groups:', groups)
                                    with open('possible_solutions.txt', 'a') as f:
                                        f.write(file_name + '\n')
                                        f.write(f'bin_thresh: {bin_thresh}\n')
                                        f.write(f'thresh_block_size: {thresh_block_size}\n')
                                        f.write(f'kernel.shape: {kernel.shape}\n')
                                        f.write(f'det_thresh: {det_thresh}\n')
                                        f.write(f'Groups: {groups}\n\n')
                                    raise FoundMatches
                            except AssertionError as e:
                                print(e)
                    except (MovedFile, AssertionError) as e:
                        print(e)
    except FoundMatches as fm:
        if groups is not None and tile_side_rotations is not None:
            pair_groups(groups, tile_side_rotations)


if __name__ == "__main__":
    file_name = 'snowflakes.jpg'
    files = False
    for f in glob.glob('square_*'):
        os.remove(f)
    main(file_name, files)
