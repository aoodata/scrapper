import numpy as np
import cv2
import imageio.v3 as iio
from utils import imshow

def imagetobase64(image):
    import imageio.v3 as iio
    import base64
    image = iio.imwrite("<bytes>", image, extension=".png")
    image = base64.b64encode(image)
    return image.decode('utf-8')

def trim_image(image, border=10, threshold=105): # border was threshold was 105
    """
    Trim an image to the minimum bounding box around non-zero pixels
    @TODO: adjust threshold for sc event images 155?
    @TODO: adjust border, reduce to 2 ?
    :param image:
    :param border:
    :param threshold:
    :return:
    """
    op = image.copy()
    op[op < threshold] = 0 #105
    #imshow(op, cmap="gray")
    coords = cv2.findNonZero(op)  # Find all non-zero points (text)
    x, y, w, h = cv2.boundingRect(coords)  # Find minimum spanning bounding box
    x = max(x - border, 0)
    y = max(y - border, 0)
    w = min(w + 2 * border, image.shape[1] - x)
    h = min(h + 2 * border, image.shape[0] - y)
    image = image[y:y + h, x:x + w]  # Crop the image - note we do this on the original image
    #imshow(image, cmap="gray")
    return image

def pad_image(image, padsize=5, pad_value='median'):
    if pad_value == "median":
        # median of the borders of image
        pad_value = np.median(np.concatenate([image[0, :], image[-1, :], image[:, 0], image[:, -1]]))

    image = np.pad(image, ((padsize, padsize), (padsize, padsize)), 'constant', constant_values=pad_value)
    return image

def image_to_grayscale(image):
    if len(image.shape) == 3:
        image = np.mean(image, axis=2).astype(np.uint8)
    return image

def rescale_image(image, scale=2):
    image = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)), interpolation=cv2.INTER_CUBIC)
    return image

def repeat_image(image, number):
    image = np.hstack([image] * number)
    return image

def find_max_pattern(image, pattern, threshold=0.85):

    try:
        match = cv2.matchTemplate(image, pattern, cv2.TM_SQDIFF_NORMED )
        # find max value location
        #imshow(image)
        #imshow(pattern)
        #imshow(match)
        minv, maxv, minloc, maxloc = cv2.minMaxLoc(match)
        if minv > 1 - threshold:
            return None
        top_left = minloc
        center = (top_left[0] + pattern.shape[1] // 2, top_left[1] + pattern.shape[0] // 2)
        return center

    except Exception as e:
        return None



def find_leftest_pattern(image, pattern, threshold=0.85):
    #imshow(image, cmap="gray")
    #imshow(pattern, cmap="gray")
    try:
        match = cv2.matchTemplate(image, pattern, cv2.TM_CCOEFF_NORMED)
        #imshow(image, cmap="gray")
        #imshow(pattern, cmap="gray")
        #imshow(match, cmap="gray")
        match_bin = (match > threshold).astype(np.uint8) * 255
        ccs = cv2.connectedComponentsWithStats(match_bin)
    except Exception as e:
        return None
    if ccs[0] == 1:
        return None
    leftest_component = np.argmin(ccs[3][1:, 0]) + 1
    return ccs[2][leftest_component, 0], ccs[2][leftest_component, 1]