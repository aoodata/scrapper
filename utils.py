import cv2
import PIL
import numpy as np
from skimage.transform import rescale
#pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
import win32api
import win32con
from pywinauto import mouse, keyboard
import pickle

#from tesseractAPI import tessAPI, tessAPILine, RIL

#import easyocr
#reader = easyocr.Reader(['en','fr'])

def first(d):
    return d[next(iter(d))]

def get_box_absolute(image, box):
    height = image.shape[0]
    width = image.shape[1]
    minbx = min(box[0][0], box[1][0])
    minby = min(box[0][1], box[1][1])
    maxbx = max(box[0][0], box[1][0])
    maxby = max(box[0][1], box[1][1])
    box = ((int(minbx * width), int(minby * height)), (int(maxbx * width), int(maxby * height)))
    return box

def get_box(image, box):
    box = get_box_absolute(image, box)
    return image[box[0][1]:box[1][1], box[0][0]:box[1][0],:]

def is_in_box(image, point, box):
    box = get_box_absolute(image, box)
    return box[0][0] <= point[0] <= box[1][0] and box[0][1] <= point[1] <= box[1][1]

def get_box_center(image, box):
    height = image.shape[0]
    width = image.shape[1]
    return int(width * (box[0][0] + box[1][0]) / 2), int(height * (box[0][1] + box[1][1]) / 2)


def click(position):
    mouse.click(button='left', coords=position)

def send_key(key):
    keyboard.send_keys(key)

def click_center(image, box):
    center = get_box_center(image, box)
    click(center)

def preproc_text_image(image, horizontal_copy=1, trim=True, padsize=20, threshold=105, pad_value=255):

    image = (255 - (np.mean(image, axis=2))).astype(np.uint8)
    image = rescale(image, 2, anti_aliasing=True)#, multichannel=False)
    image = (255 * image).astype(np.uint8)
    if trim:
        image = 255-trim_image(255-image, border=3, threshold=threshold)
    #imshow(image, cmap="gray")
    if horizontal_copy > 1:
        image = np.hstack([image] * horizontal_copy)
    if trim:
        image = pad_image(image, padsize=padsize, pad_value=pad_value)
        #if pad_value == "median":
        #    # median of the borders of image
        #    pad_value = np.median(np.concatenate([image[0, :], image[-1, :], image[:, 0], image[:, -1]]))
        ##pad image 10 pixels on each side
        #image = np.pad(image, ((padsize, padsize), (padsize, padsize)), 'constant', constant_values=pad_value)
    return image

#def extract_text_easyocr(image, mode='line', customallows=None):
#    allowlist = None
#    if mode == 'line':
#        ...
#    elif mode == 'word':
#        ...
#    elif mode == 'digits':
#        allowlist = '0123456789,'
#    elif mode == 'all':
#        ...
#    else:
#        raise ValueError('Mode not recognized')
#
#    if customallows is not None:
#        allowlist = customallows
#    #image = cv2.resize(image, (0, 0), fx=1.5, fy=1.5, interpolation=cv2.INTER_AREA)
#    #if len(image.shape) == 3:
#    #    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#    #image = 255 - image
#    #imshow(image, cmap="gray")
#    #data = pytesseract.image_to_string(
#    #    image,
#    #    lang="eng",
#    #    config=config)
#    # image as PIL image
#    #api.SetImage(PIL.Image.fromarray(image))
#    #if mode == 'digits':
#    #    api.SetVariable("tessedit_char_whitelist", "0123456789")
#    #else:
#    #    api.SetVariable("tessedit_char_whitelist", "")
#    #data = api.GetUTF8Text()
#    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#    image = cv2.resize(image, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_AREA)
#    data = reader.readtext(image, allowlist=allowlist, detail=0)
#
#    if len(data) == 0:
#
#        return ""
#
#    #print(data)
#    return data[0]



def imshow(image, click_event=None, cmap=None):
    """
    Show an image at true scale
    """
    import matplotlib.pyplot as plt
    dpi = 80
    margin = 0.5  # (5% of the width/height of the figure...)
    h, w = image.shape[:2]

    # Make a figure big enough to accomodate an axis of xpixels by ypixels
    # as well as the ticklabels, etc...
    figsize = (1 + margin) * w / dpi, (1 + margin) * h / dpi

    fig = plt.figure(figsize=figsize, dpi=dpi)
    # Make the axis the right size...
    ax = fig.add_axes([0, 0, 1, 1])

    ax.imshow(image, interpolation='none', cmap=cmap)

    plt.axis('off')
    plt.show()

    return fig, ax

def center(window):
    r = window.rectangle()
    return (int(r.left + r.width() / 2), int(r.top + r.height() / 2))



def match_image_name2(image1, image2, thr=0.85, return_score=False, size_diff_thr=30):
    # flatten background of the image
    image1 = image1.copy()
    image2 = image2.copy()


    image2 = trim_image(255 - image2)#[:, 20:-20])
    image1 = trim_image(255 - image1)#[:, 20:-20])

    #imshow(image1, cmap="gray")
    #imshow(image2, cmap="gray")
    #image1[image1 < 105] = 0  # 105
    #image2[image2 < 105] = 0  # 105

    if abs(image1.shape[1] - image2.shape[1]) > size_diff_thr:
        return -1 if return_score else False

    image1 = cv2.copyMakeBorder(image1, 20, 20, 10, 10, cv2.BORDER_CONSTANT, value=0)

    # if dimensions are too different, don't even try

    if image1.shape[0] >= image2.shape[0] and image1.shape[1] >= image2.shape[1]:
        try:
            m = cv2.matchTemplate(image1, image2, cv2.TM_CCOEFF_NORMED).max()
        except Exception as e:
            raise Exception("Error in matchTemplate", e)
        return m if return_score else m > thr
    else:
        return -1 if return_score else False

def template_match(image1, image2):


    image1 = cv2.copyMakeBorder(image1, 20, 20, 10, 10, cv2.BORDER_CONSTANT, value=0)

    # if dimensions are too different, don't even try

    if image1.shape[0] >= image2.shape[0] and image1.shape[1] >= image2.shape[1]:
        try:
            m = cv2.matchTemplate(image1, image2, cv2.TM_CCOEFF_NORMED).max()
        except Exception as e:
            raise Exception("Error in matchTemplate", e)
        return m
    else:
        return 0


def skeleton_match(image1, image2):
    import higra as hg
    from skimage.morphology import skeletonize
    image1 = skeletonize(image1 > 105).astype(np.uint8)
    image2 = skeletonize(image2 > 105).astype(np.uint8)

    image1 = trim_image(image1, 0, 0)
    image2 = trim_image(image2, 0, 0)

    max_width = max(image1.shape[1], image2.shape[1])
    max_height = max(image1.shape[0], image2.shape[0])

    def increase_image_size(im):
        top = (max_height - im.shape[0]) // 2
        bottom = max_height - im.shape[0] - top
        left = (max_width - im.shape[1]) // 2
        right = max_width - im.shape[1] - left
        return cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

    image1 = increase_image_size(image1)
    image2 = increase_image_size(image2)

    # imshow(image1, cmap="gray")
    # imshow(image2, cmap="gray")
    try:
        match1, match2 = hg.match_pixels_image_2d(image1, image2, max_distance=3, mode="absolute")
        score = match1.size / max(np.count_nonzero(image1), np.count_nonzero(image2))
        return score
    except Exception as e:
        print("Error in skeleton_match", e)
        return 0


def match_image_name(image1, image2, thr=0.85, return_score=False, size_diff_thr=30, rescale=True):

    # flatten background of the image
    image1 = image1.copy()
    image2 = image2.copy()

    image2 = trim_image(255 - image2)#[:, 20:-20])
    image1 = trim_image(255 - image1)#[:, 20:-20])

    #imshow(image1, cmap="gray")
    #imshow(image2, cmap="gray")
    #image1[image1 < 105] = 0  # 105
    #image2[image2 < 105] = 0  # 105

    #if abs(image1.shape[1] - image2.shape[1]) > size_diff_thr:
    #    return -1 if return_score else False
    if min(image1.shape[1], image2.shape[1]) / max(image1.shape[1], image2.shape[1]) < 0.8:
        return -1 if return_score else False
    if rescale:
        if image1.shape[1] * image1.shape[1] > image2.shape[1] * image2.shape[1]:
            image1 = cv2.resize(image1, (image2.shape[1], image2.shape[0]), interpolation=cv2.INTER_AREA)
        else:
            image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]), interpolation=cv2.INTER_AREA)

    score1 = template_match(image1, image2)
    score2 = skeleton_match(image1, image2)
    score = (score1 + score2) / 2
    #score = float(max(score1, score2))
    return score if return_score else score > thr

def match_image_name_saved(image1, image2, thr=0.85, return_score=False, size_diff_thr=30):

    # flatten background of the image
    image1 = image1.copy()
    image2 = image2.copy()

    image2 = trim_image(255 - image2)#[:, 20:-20])
    image1 = trim_image(255 - image1)#[:, 20:-20])

    #imshow(image1, cmap="gray")
    #imshow(image2, cmap="gray")
    #image1[image1 < 105] = 0  # 105
    #image2[image2 < 105] = 0  # 105

    if abs(image1.shape[1] - image2.shape[1]) > size_diff_thr:
        return -1 if return_score else False

    score1 = template_match(image1, image2)
    score2 = skeleton_match(image1, image2)
    score = (score1 + score2) / 2

    return score if return_score else score > thr


#def remove_alliance_name(image):
#
#    print("remove_alliance_name is deprecated")
#    tessAPILine.SetImage(PIL.Image.fromarray(image))
#    boxes = tessAPILine.GetComponentImages(RIL.WORD, True)
#    if len(boxes) <= 1:
#        return image
#    alliance_name = boxes[0][1]
#    image = image.copy()
#    image[alliance_name['y']-2:alliance_name['y'] + alliance_name['h']+4, alliance_name['x']-2:alliance_name['x'] + alliance_name['w']+4] = 255
#    return image

import imageio.v3 as imageio

#par_left_one_line = 255 - imageio.imread('patterns/parenthesis_left_one_line.png').mean(axis=2).astype(np.uint8)
#par_right_one_line = 255 - imageio.imread('patterns/parenthesis_right_one_line.png').mean(axis=2).astype(np.uint8)
#
#par_left_two_lines = 255 - imageio.imread('patterns/parenthesis_left_two_lines.png').mean(axis=2).astype(np.uint8)
#par_right_two_lines = 255 - imageio.imread('patterns/parenthesis_right_two_lines.png').mean(axis=2).astype(np.uint8)

par_left_one_line = imageio.imread('screenshots/par_left2.png').mean(axis=2).astype(np.uint8)
par_right_one_line = imageio.imread('screenshots/par_right2.png').mean(axis=2).astype(np.uint8)

par_left_two_lines = imageio.imread('screenshots/par_left.png').mean(axis=2).astype(np.uint8)
par_right_two_lines = imageio.imread('screenshots/par_right.png').mean(axis=2).astype(np.uint8)


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



#def find_leftest_pattern(image, pattern, threshold=0.85):
#    #imshow(image, cmap="gray")
#    #imshow(pattern, cmap="gray")
#    try:
#        match = cv2.matchTemplate(image, pattern, cv2.TM_CCOEFF_NORMED)
#        match_bin = (match > threshold).astype(np.uint8) * 255
#        ccs = cv2.connectedComponentsWithStats(match_bin)
#    except Exception as e:
#        return None
#    if ccs[0] == 1:
#        return None
#    leftest_component = np.argmin(ccs[3][1:, 0]) + 1
#    return ccs[2][leftest_component, 0], ccs[2][leftest_component, 1]

#def detect_alliance_box(image, two_lines=False):
#    #image = 255 - image
#    par_left = par_left_two_lines if two_lines else par_left_one_line
#    par_right = par_right_two_lines if two_lines else par_right_one_line
#    matchleft = find_leftest_pattern(image, par_left)
#    if matchleft is None:
#        return None
#    matchright = find_leftest_pattern(image, par_right)
#    if matchright is None:
#        return None
#    outer_box = (matchleft[0], matchleft[1], matchright[0] + par_right.shape[1], matchright[1] + par_right.shape[0])
#    inner_box = (matchleft[0] + par_left.shape[1], matchleft[1], matchright[0] + 1, matchright[1] + par_right.shape[0])
#    return {'outer': outer_box, 'inner': inner_box}


#def remove_alliance_name_image(image, two_lines=False):
#    box = detect_alliance_box(image, two_lines=two_lines)
#    if box is None:
#        return image
#    image = image.copy()
#    extra = 4 if two_lines else 5
#    outer_box = box['outer']
#    minx, miny = outer_box[:2]
#    maxx, maxy = outer_box[2:]
#    minx = max(0, minx - extra)
#    miny = max(0, miny - extra)
#    maxx = min(image.shape[1], maxx + extra)
#    maxy = min(image.shape[0], maxy + extra)
#    image[miny:maxy, minx:maxx] = 255
#    return image

# adapted for two lines text
def remove_alliance_name2(image):

    matchleft = cv2.matchTemplate(image, par_left_two_lines, cv2.TM_CCOEFF_NORMED)
    _, maxv, _, maxlocleft = cv2.minMaxLoc(matchleft)
    if maxv < 0.85:
        return image

    matchright = cv2.matchTemplate(image, par_right_two_lines, cv2.TM_CCOEFF_NORMED)
    _, maxv, _, maxlocright = cv2.minMaxLoc(matchright)
    if maxv < 0.85:
        return image

    extra = 4
    minx, miny = maxlocleft
    maxx, maxy = maxlocright[0] + par_right_two_lines.shape[1], maxlocright[1] + par_right_two_lines.shape[0] - 2
    minx = max(0, minx - extra)
    miny = max(0, miny - extra)
    maxx = min(image.shape[1], maxx + extra)
    maxy = min(image.shape[0], maxy + extra)
    image = image.copy()
    image[miny:maxy, minx:maxx] = 255
    return image


# adapted for one line text
def remove_alliance_name3(image):

    try:
        matchleft = cv2.matchTemplate(image, par_left_one_line, cv2.TM_CCOEFF_NORMED)
        _, maxv, _, maxlocleft = cv2.minMaxLoc(matchleft)
        if maxv < 0.85:
            return image

        matchright = cv2.matchTemplate(image, par_right_one_line, cv2.TM_CCOEFF_NORMED)
        _, maxv, _, maxlocright = cv2.minMaxLoc(matchright)
        if maxv < 0.85:
            return image
    except Exception as e:
        return image
    extra = 5
    minx, miny = maxlocleft
    maxx, maxy = min(maxlocright[0] + par_right_one_line.shape[1] + 5, image.shape[1] ), min(maxlocright[1] + par_right_one_line.shape[0] + 5, image.shape[0])
    miny = max(0, miny - extra)
    maxy = min(image.shape[0], maxy + extra)
    image = image.copy()
    image[miny:maxy, minx:maxx] = 255
    return image



class SortedFixSizedList:

    def __init__(self, max_size):
        self.max_size = max_size
        self.data = []

    def add(self, key, item):
        for i, (k, v) in enumerate(self.data):
            if v == item:
                if key > k:
                    self.data[i] = (key, item)
                    self.data.sort(key=lambda x: x[0])
                return
        if len(self.data) < self.max_size:
            self.data.append((key, item))
            self.data.sort(key=lambda x: x[0])
        else:
            if key > self.data[0][0]:
                self.data[0] = (key, item)
                self.data.sort(key=lambda x: x[0])

def mouse_scroll(window, dist, position=None):
    window.set_focus()
    if position is None:
        position = center(window)
    mouse.move(coords=position)
    win32api.mouse_event(win32con.MOUSEEVENTF_WHEEL | win32con.MOUSEEVENTF_ABSOLUTE, position[0], position[1], dist)

def capture_window(window, initial_sleep=0, validation_function=None, sleep=0.1, max_attempts=20):
    import time
    if initial_sleep > 0:
        time.sleep(initial_sleep)
    window.set_focus()
    for i in range(max_attempts):

        image = np.array(window.capture_as_image())
        if validation_function is None or validation_function(image):
            return image

        time.sleep(sleep)
    raise Exception("Could not capture window")

###################################################
# Alliance name removal
###################################################

resize_model_regressor_filename = "resizing_decision_tree.pkl"
resize_model_regressor = None
# read model with pickle

with open(resize_model_regressor_filename, 'rb') as f:
    resize_model_regressor = pickle.load(f)

def trim_image_alliance_removal(image):
    op = image.copy()
    #imshow(op, cmap="gray")
    op[op > 150] = 0 #105
    #imshow(op, cmap="gray")
    coords = cv2.findNonZero(op)  # Find all non-zero points (text)
    x, y, w, h = cv2.boundingRect(coords)  # Find minimum spanning bounding box
    x = max(x - 5, 0)
    y = max(y - 5, 0)
    w = min(w + 10, image.shape[1] - x)
    h = min(h + 10, image.shape[0] - y)
    image = image[y:y + h, x:x + w]  # Crop the image - note we do this on the original image
    #imshow(image)
    return image

def proc_image_with_alliance(image, rescale=True):

    if has_two_lines_alliance_removal(image): #image.shape[0] >= 47: # long name on two lines...
        #image2 = trim_image_alliance_removal(remove_alliance_name_image(image, two_lines=True))
        image2 = trim_image_alliance_removal(remove_alliance_name2(image))

        if rescale:
            factors = resize_model_regressor.predict(
                [[image2.shape[0], image2.shape[1], image2.shape[0] / image2.shape[1]]])
            factors = factors[0]
            return cv2.resize(image2, (int(image2.shape[1] * factors[0]), int(image2.shape[0] * factors[1])))
        else:
            return image2
    else:
        #image2 = trim_image_alliance_removal(remove_alliance_name_image(image,two_lines=False))
        image2 = trim_image_alliance_removal(remove_alliance_name3(image))
        image2 = image2[:-3, 3:]

        if rescale:
            factors = resize_model_regressor.predict(
                [[image2.shape[0], image2.shape[1], image2.shape[0] / image2.shape[1]]])
            factors = factors[0]
            return cv2.resize(image2, (int(image2.shape[1] * factors[0]), int(image2.shape[0] * factors[1])))
        else:
            return image2



def has_two_lines_alliance_removal(image):
    #imshow(image, cmap="gray")
    max_lines = image.min(axis=1)
    #imshow(max_lines[:,None], cmap="gray")
    max_lines = max_lines > 140
    max_lines = max_lines.astype(np.uint8)
    ccs = cv2.connectedComponentsWithStats(max_lines)
    if ccs[0] < 3:
        return False
    #height between lines
    ccs_y = ccs[3][:,1]
    heights = ccs_y[1:] - ccs_y[:-1]
    count = np.count_nonzero(heights > 15)


    return count >= 2 # background, bottom line, top line

###################################################
# Name image parsing
###################################################

#def parse_name_image(image):
#
#    two_lines = has_two_lines_alliance_removal(image)
#    boxes = detect_alliance_box(image, two_lines=two_lines)
#
#    if boxes is None and two_lines is False:
#        boxes = detect_alliance_box(image, two_lines=True)
#
#    if boxes is None:
#        return "", extract_text_easyocr(image)
#    alliance_box = boxes['inner']
#
#    alliance_image = image[alliance_box[1]:alliance_box[3], alliance_box[0]:(alliance_box[2]-2)]
#    #alliance_image = np.pad(alliance_image, ((5, 5), (5, 5)), 'constant', constant_values=255)
#    if alliance_image.shape[0] < 10 or alliance_image.shape[1] < 10:
#        alliance = ""
#    else:
#        alliance_image = pad_image(alliance_image, padsize=10)
#        #imshow(alliance_image, cmap="gray")
#        alliance = extract_text_easyocr(alliance_image)
#
#    name_image = trim_image_alliance_removal(remove_alliance_name_image(image, two_lines=two_lines))
#    #imshow(name_image, cmap="gray")
#    name = extract_text(name_image)
#
#    return alliance, name

########################################################
# frenzy and void dates
########################################################
from datetime import date, timedelta
referenceStrongestCommanderEventDate = date(2023, 1, 29)
referenceStrongestCommanderEventType = "void"
referenceStrongestCommanderEventNotType = "frenzy"

def getStrongestCommanderEventFromCycleNumber(cycle_number):
        days = cycle_number * 14
        date = referenceStrongestCommanderEventDate + timedelta(days=days)
        return {
            "date": date,
            "type": referenceStrongestCommanderEventType if (cycle_number % 2 == 0) else referenceStrongestCommanderEventNotType,
            "cycle_number": cycle_number,
        }

def get_date_of_first_strongest_commander_event_before(ref_date=None, event_type="any"):

    if ref_date is None:
        ref_date = date.today()

    # number of days between date and the reference event date
    days = (ref_date - referenceStrongestCommanderEventDate).days
    # number of events between date and the reference event date
    events = days // 14
    # number of days between the reference event date and the event we want to find
    days_before = events * 14
    # date of the event we want to find
    event_date = referenceStrongestCommanderEventDate + timedelta(days=days_before)
    # type of the event we want to find
    eventType = "void" if events % 2 == 0 else "frenzy"

    if event_type == "any":
        return {"date": event_date, "type": eventType, "cycle_number": events}
    elif event_type == eventType:
        return {"date": event_date, "type": event_type, "cycle_number": events}
    else:
        # 14 days before the event we found
        previous_event_date = event_date - timedelta(days=14)
        return {"date": previous_event_date, "type": event_type, "cycle_number": events}


collection_type_id ={
    "commander_power": 1,
    "commander_kill": 2,
    "commander_city": 3,
    "commander_officer": 4,
    "commander_titan": 5,
    "commander_island": 6,
    "commander_merit": 7,
    "commander_level": 8,
    "alliance_power": 9,
    "alliance_kill": 10,
    "alliance_territory": 11,
    "alliance_elite": 12,
    "alliance_mines": 13,
    "alliance_benefits": 14,
    "commander_ke_frenzy": 15,
    "commander_sc_frenzy": 16,
    "commander_ke_void": 17,
    "commander_sc_void": 18,
    "commander_warplane": 23,
    "commander_reputation": 24,
    "commander_loss": 25,
}

def date_to_timestamp(d):
    import time
    return time.mktime(d.timetuple())


def timestamp_to_date(ts):
    import datetime
    return datetime.datetime.fromtimestamp(ts)

def get_collection_id(cursor, collection_name, collection_timestamp, before, return_date=False):
    collection_id = collection_type_id[collection_name]
    if before:
        cursor.execute("select id, date from data_collections where type_id = ? and date < ? order by date desc limit 1", (collection_id, collection_timestamp))
        before = cursor.fetchone()
        return before if return_date else before[0]
    else:
        cursor.execute("select id, date from data_collections where type_id = ? and date > ? order by date asc limit 1", (collection_id, collection_timestamp))
        after = cursor.fetchone()
        return after if return_date else after[0]



##############################  IMAGE PROCESSING  ########################################
def imagetobase64(image):
    import imageio.v3 as iio
    import base64
    image = iio.imwrite("<bytes>", image, extension=".png")
    image = base64.b64encode(image)
    return image.decode('utf-8')

def trim_image(image, border=10, threshold=105):
    op = image.copy()
    op[op < threshold] = 0 #105
    coords = cv2.findNonZero(op)  # Find all non-zero points (text)
    x, y, w, h = cv2.boundingRect(coords)  # Find minimum spanning bounding box
    x = max(x - border, 0)
    y = max(y - border, 0)
    w = min(w + 2 * border, image.shape[1] - x)
    h = min(h + 2 * border, image.shape[0] - y)
    image = image[y:y + h, x:x + w]  # Crop the image - note we do this on the original image
    return image

def pad_image(image, padsize=5, pad_value='median'):
    if pad_value == "median":
        # median of the borders of image
        pad_value = np.median(np.concatenate([image[0, :], image[-1, :], image[:, 0], image[:, -1]]))

    image = np.pad(image, ((padsize, padsize), (padsize, padsize)), 'constant', constant_values=pad_value)
    return image

##########################################################################################

if __name__ == "__main__":
    d = date(2024, 1, 13)
    print(get_date_of_first_strongest_commander_event_before(d))
    print(get_date_of_first_strongest_commander_event_before(d,event_type="frenzy"))
    print(get_date_of_first_strongest_commander_event_before(d,event_type="void"))