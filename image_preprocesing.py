import cv2
import numpy as np


def preprocess_image(img, dilate=True):
    """
    Perform blurring, adaptive thresholding, inversion of colors and dilation
    to highlight the grid lines and numbers
    """
    # blur the image
    img = cv2.GaussianBlur(img.copy(), (9, 9), 0)

    # adaptive threshold to generate binary image
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 11, 2)

    # invert colors so that grid lines and numbers have non-zero pixel values
    img = cv2.bitwise_not(img)

    # dilate image to increase size of grid lines and numbers
    if dilate:
        kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]], dtype=np.uint8)
        img = cv2.dilate(img, kernel=kernel, iterations=1)

    return img


def get_grid_corners(img):
    """Returns the coordinates of the four corners of the grid."""
    contours = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    grid = max(contours, key=cv2.contourArea)
    top_left = min(grid, key=lambda p: p[0][0] + p[0][1])[0]
    top_right = max(grid, key=lambda p: p[0][0] - p[0][1])[0]
    bot_right = max(grid, key=lambda p: p[0][0] + p[0][1])[0]
    bot_left = min(grid, key=lambda p: p[0][0] - p[0][1])[0]

    return np.array([top_left, top_right, bot_right, bot_left])


def crop_and_warp(img, corners):
    """Crops and warps a rectangular section from the image into a square."""
    # longest side in the rectangle (np.linalg.norm gives euclidean distance)
    side = max(np.linalg.norm(corners[0] - corners[1]),
               np.linalg.norm(corners[1] - corners[2]),
               np.linalg.norm(corners[2] - corners[3]),
               np.linalg.norm(corners[3] - corners[0]))
    src = corners.astype(np.float32, copy=False)
    dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, matrix, (int(side), int(side)))

    return warped, matrix, corners


def divide_into_cells(img):
    """Divides the grid into 81 cells."""
    cells, points = [], []
    side = img.shape[0] // 9
    for j in range(9):
        for i in range(9):
            tl = (i * side, j * side)  # top left corner of cell
            br = ((i + 1) * side, (j + 1) * side)   # bottom right corner of cell
            cells.append((tl, br))

    return cells


def find_largest_feature(cell, scan_tl=None, scan_br=None):
    """
    Uses `cv2.floodFill` to fill the largest feature (digit)
    with white color. Fills the rest of the cell with black.
    Args:
        cell (np.array): cell of sudoku which might contain a digit
        scan_tl (list): coordinates of top-left corner of search area
        scan_br (list): coordinates of bot-right corner of search area
    Returns:
        tuple: modified cell, bounding box, seed point for flood fill
    """
    cell = cell.copy()
    height, width = cell.shape[:2]
    max_area = 0
    seed_point = (None, None)
    if not scan_tl:
        scan_tl = [0, 0]
    if not scan_br:
        scan_br = [width, height]

    # loop through the area inside cell
    for x in range(scan_tl[0], scan_br[0]):
        for y in range(scan_tl[1], scan_br[1]):
            # only run on white pixels
            if cell.item(y, x) == 255:
                area = cv2.floodFill(cell, mask=None, seedPoint=(x, y), newVal=64)
                if area[0] > max_area:
                    max_area = area[0]
                    seed_point = (x, y)

    # color all pixels that are white to gray
    for x in range(width):
        for y in range(height):
            if cell.item(y, x) == 255:
                cv2.floodFill(cell, mask=None, seedPoint=(x, y), newVal=64)

    mask = np.zeros((height + 2, width + 2), np.uint8)

    # highlight the main feature (digit)
    if seed_point[0] is not None and seed_point[1] is not None:
        cv2.floodFill(cell, mask=mask, seedPoint=seed_point, newVal=255)

    top, bot, left, right = height, 0, width, 0

    for x in range(width):
        for y in range(height):
            # hide (black) anything that isn't main feature
            if cell.item(y, x) == 64:
                cv2.floodFill(cell, mask=mask, seedPoint=(x, y), newVal=0)

            # find bounding box
            if cell.item(y, x) == 255:
                top = min(top, y)
                bot = max(bot, y)
                left = min(left, x)
                right = max(right, x)

    bbox = [[left, top], [right, bot]]

    return cell, np.array(bbox, dtype=np.float32), seed_point


def scale_and_centre(img, size, margin=0, background=0):
    """Scales and centres an image onto a new background cell."""
    h, w = img.shape[:2]

    def centre_pad(length):
        """Handle centering for a given length that may be odd or even."""
        if length % 2 == 0:
            side1 = (size - length) // 2
            side2 = side1
        else:
            side1 = (size - length) // 2
            side2 = side1 + 1

        return side1, side2

    if h > w:
        t_pad = b_pad = margin // 2
        ratio = (size - margin) / h
        w, h = int(ratio * w), int(ratio * h)
        l_pad, r_pad = centre_pad(w)
    else:
        l_pad = r_pad = margin // 2
        ratio = (size - margin) / w
        w, h = int(ratio * w), int(ratio * h)
        t_pad, b_pad = centre_pad(h)

    img = cv2.resize(img, (w, h))
    img = cv2.copyMakeBorder(img, t_pad, b_pad, l_pad, r_pad,
                             cv2.BORDER_CONSTANT, None, background)

    return cv2.resize(img, (size, size))


def extract_digit(img, cell, size):
    """Extracts a digit from the cell if present."""
    digit = img[cell[0][1]:cell[1][1], cell[0][0]:cell[1][0]]
    h, w = digit.shape[:2]

    # margin is the area inside the cell where the digit would be found
    margin = int(np.mean((h, w)) / 2.6)
    bbox = find_largest_feature(digit, [margin, margin], [w - margin, h - margin])[1]
    digit = digit[int(bbox[0][1]):int(bbox[1][1]), int(bbox[0][0]):int(bbox[1][0])]

    # scale and pad the digit
    w = bbox[1][0] - bbox[0][0]
    h = bbox[1][1] - bbox[0][1]

    # ignore small bounding boxes
    if w > 0 and h > 0 and w * h > 100 and len(digit) > 0:
        return scale_and_centre(digit, size, margin=4)
    return np.zeros((size, size), np.uint8)


def get_digits(img, cells, size):
    """Returns an array of extracted digits from the cells."""
    img = preprocess_image(img.copy(), dilate=False)

    return [extract_digit(img, cell, size) for cell in cells]


def display_digits(digits, color=255):
    """Displays list of 81 digits in grid format."""
    rows = []
    with_border = [cv2.copyMakeBorder(
        img.copy(), 1, 1, 1, 1, cv2.BORDER_CONSTANT, None, color) for img in digits]
    for i in range(9):
        row = np.concatenate(with_border[i * 9:(i + 1) * 9], axis=1)
        rows.append(row)

    return np.concatenate(rows)


def scale_down(img, max_size):
    """Scales down image to `max_size`x`max_size` if the image is larger than max size."""
    height, width = img.shape[:2]
    ratio = 1
    max_dim = max(height, width)
    if max_dim > max_size:
        ratio = max_size / max_dim

    return cv2.resize(img, (int(ratio * width), int(ratio * height)))


def crop_grid(img):
    """Crops the image and returns only the sudoku grid."""
    processed = preprocess_image(img)
    corners = get_grid_corners(processed)
    return crop_and_warp(img, corners)


def convert(img):
    """Divides the image into 81 images of individual cells."""
    original = img.copy()
    processed = preprocess_image(original)
    corners = get_grid_corners(processed)
    cropped = crop_and_warp(original, corners)[0]
    cells = divide_into_cells(cropped)
    return get_digits(cropped, cells, size=28)