#from tesseractAPI import tessAPI, tessAPILine

import numpy as np
import easyocr
import uuid
import os
import shutil
import json
import imageio
import time
import numpy as np
import aooutils.image as imgu
from utils import imshow
import cv2
import imageio.v3 as imageio

reader = easyocr.Reader(['en','fr'])

class AOOErrorOCR(Exception):

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class Reader:
    _record = False
    _record_path = "c:/ocr_training/"
    _record_image_path = _record_path + "images/"
    _record_metadata_file = _record_path + "data.json"
    _record_metadata = None

    def __init__(self, description):
        self.description = description

    @classmethod
    def start_record(cls):
        cls._record = True
        # if metadata file exists, load it
        if os.path.exists(cls._record_metadata_file):
            with open(cls._record_metadata_file, 'r') as f:
                cls._record_metadata = json.load(f)
        else:
            cls._record_metadata = {}

    @classmethod
    def stop_record(cls):
        cls._record = False
        cls._save_record()

    @classmethod
    def _save_record(cls):
        if cls._record_metadata is not None:
            with open(cls._record_metadata_file, 'w') as f:
                json.dump(cls._record_metadata, f)

    def read(self, image, *, tag, **kwargs) -> int | str:
        if self._record:

            result, preprocessed_image, ocr_result = self._read(image, tag, **kwargs)

            if ocr_result is not None:
                record_id = None
                while record_id is None or record_id in self._record_metadata:
                    record_id = str(uuid.uuid4())
                record_image_path = self._record_image_path + record_id + "_input.png"
                imageio.imwrite(record_image_path, image)
                record_preprocessed_image_path = self._record_image_path + record_id + "_preprocessed.png"
                imageio.imwrite(record_preprocessed_image_path, preprocessed_image)
                metadata = {'id': record_id, 'context': self.description, 'tag': tag, 'time': time.time(), 'kwargs': kwargs, 'result': result, 'ocr_result': ocr_result}
                self._record_metadata[record_id] = metadata
                if len(self._record_metadata) % 100 == 0:
                    self._save_record()
            return result

        else:
            return self._read(image, tag, **kwargs)

    def _read(self, image, tag, **kwargs):
        raise NotImplementedError("Subclass must implement abstract method")

    def _readtext(self, image, mode, engine='easyocr', allowed_chars=None):
        if engine == 'tesseract':
            raise NotImplementedError("Tesseract not supported")
            #return self._readtext_tesseract(image, mode)
        elif engine == 'easyocr':
            return self._readtext_easyocr(image, mode, allowed_chars)

    # def _readtext_tesseract(self, image, mode=None, allowed_chars=None):
    #     from PIL import Image as PILImage
    #     if mode == 'line':
    #         api = tessAPILine
    #         api.SetVariable("tessedit_char_whitelist", "")
    #     elif mode == 'word':
    #         api = tessAPILine
    #         api.SetVariable("tessedit_char_whitelist", "")
    #     elif mode == 'digits':
    #         api = tessAPILine
    #         api.SetVariable("tessedit_char_whitelist", "0123456789")
    #     elif mode is None or mode == 'all':
    #         api = tessAPI
    #         api.SetVariable("tessedit_char_whitelist", "")
    #     else:
    #         raise ValueError('Mode not recognized')
    #
    #     if allowed_chars is not None:
    #         api.SetVariable("tessedit_char_whitelist", allowed_chars)
    #
    #     api.SetImage(PILImage.fromarray(image))
    #     data = api.GetUTF8Text()
    #
    #     return data[:-1]

    def _readtext_easyocr(self, image, mode=None, allowed_chars=None):
        allowlist = None
        if mode == 'line':
            ...
        elif mode == 'word':
            ...
        elif mode == 'digits':
            allowlist = '0123456789,'
        elif mode is None or mode == 'all':
            ...
        else:
            raise ValueError('Mode not recognized')

        if allowed_chars is not None:
            allowlist = allowed_chars

        #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        #image = cv2.resize(image, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_AREA)
        #data = reader.readtext(image, allowlist=allowlist, detail=0)
        data = reader.recognize(image, allowlist=allowlist, detail=0)
        if len(data) == 0:
            return ""

        return "".join(data)


def preprocess_image(image, *, invert=False, scale=2, padsize=10, duplications=0, threshold=140, border=2):
    image = imgu.image_to_grayscale(image)
    if invert:
        image = 255 - image
    image = imgu.trim_image(image, border=border, threshold=threshold)
    #imshow(image)
    if duplications:
        image = imgu.repeat_image(image, duplications)
    if scale > 1:
        image = imgu.rescale_image(image, scale=scale)
    if padsize > 0:
        image = imgu.pad_image(image, padsize=padsize, pad_value='median')
    image = 255 - image
    return image

class ReaderLargeNumbers(Reader):

    def __init__(self):
        super().__init__(type(self).__name__)

    def _read(self, image, tag, **kwargs):
        engine = kwargs.get('engine', 'easyocr')
        invert = kwargs.get('invert', False)
        threshold = kwargs.get('threshold', 140)
        #imshow(image, cmap="gray")
        image = preprocess_image(image, invert=invert, scale=3, padsize=20, duplications=0, threshold=threshold)
        #imshow(image, cmap="gray")
        value_txt = self._readtext(image, mode='digits', engine=engine, allowed_chars='0123456789,')
        value_txt = ''.join([i for i in value_txt if i.isdigit()])
        if len(value_txt) == 0:
            value = 0
        else:
            value = int(value_txt)

        if self._record:
            #format value with a comma every 3 digits
            gt = '{:,}'.format(value)
            return value, image, gt
        else:
            return value


class ReaderText(Reader):

        def __init__(self):
            super().__init__(type(self).__name__)

        def _read(self, image, tag,  **kwargs):
            engine = kwargs.get('engine', 'easyocr')
            mode = kwargs.get('mode', 'all')
            invert = kwargs.get('invert', False)
            duplications = kwargs.get('duplications', 0)
            scale = kwargs.get('scale', 3)
            threshold = kwargs.get('threshold', 140)

            border = 2 if duplications == 0 else 4

            image = preprocess_image(image, invert=invert, scale=scale, padsize=10, duplications=duplications, border=border, threshold=threshold)
            #imshow(image, cmap="gray")
            value_txt = self._readtext(image, mode=mode, engine=engine)
            if duplications > 0:
                if len(value_txt) % duplications != 0 or len(value_txt) == 0:
                    raise AOOErrorOCR("Error reading value: '" + value_txt + "'")
                value_final = value_txt[0:len(value_txt) // duplications]
            else:
                value_final = value_txt

            if self._record:
                return value_final, image, value_txt
            else:
                return value_final


class ReaderSmallNumbers(Reader):
    def __init__(self):
        super().__init__(type(self).__name__)

    def _read(self, image, tag, **kwargs):
        engine = kwargs.get('engine', 'easyocr')
        duplications = kwargs.get('duplications', 5)
        scale = kwargs.get('scale', 3)
        invert = kwargs.get('invert', False)
        padsize = kwargs.get('padsize', 10)

        image = preprocess_image(image, invert=invert, scale=scale, padsize=padsize, duplications=duplications, border=2)

        value_txt = self._readtext(image, mode='digits', engine=engine, allowed_chars='0123456789')
        if len(value_txt) % duplications != 0 or len(value_txt) == 0:
            imshow(image, cmap="gray")
            raise AOOErrorOCR("Error reading value: '" + value_txt + "'")
            #imshow(value)
            #print("Error reading value", value_txt, " input value:")
            #value_txt = int(input())

        value = int(value_txt[0:len(value_txt) // duplications])

        if self._record:
            return value, image, value_txt
        else:
            return value




class ReaderRank(Reader):

    def __init__(self):
        super().__init__(type(self).__name__)

        self._first_rank_mean = np.array([185.58750, 153.07250, 79.41250])
        self._second_rank_mean = np.array([150.15250, 170.73250, 166.80000])
        self._third_rank_mean = np.array([145.25500, 106.68000, 86.75000])
        self._small_numbers_reader = ReaderSmallNumbers()

    def _read(self, image, tag, **kwargs):
        engine = kwargs.get('engine', 'easyocr')
        duplications = kwargs.get('duplications', 5)

        rank_center = (image.shape[1] // 2, image.shape[0] // 2)
        rank_mean = np.mean(
            image[rank_center[1] - 10:rank_center[1] + 10, rank_center[0] - 10:rank_center[0] + 10, :], axis=(0, 1))
        value = None
        if np.linalg.norm(rank_mean - self._first_rank_mean) < 20:
            value = 1
        elif np.linalg.norm(rank_mean - self._second_rank_mean) < 20:
            value = 2
        elif np.linalg.norm(rank_mean - self._third_rank_mean) < 20:
            value = 3

        if value is None:
            value = self._small_numbers_reader.read(image, tag=tag, engine=engine, duplications=duplications, scale=1, padsize=5)

        if self._record:
            return value, None, None
        else:
            return value


class ReaderMeritProgression(Reader):

        def __init__(self):
            super().__init__(type(self).__name__)
            self._merit_text_color = np.array([255, 240, 196])

        def _read(self, image, tag, **kwargs):
            engine = kwargs.get('engine', 'easyocr')

            mask = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mask = mask - self._merit_text_color
            mask = np.sum(np.abs(mask), axis=2) / 3
            mask = mask > 70 # 70

            image[mask, :] = [0, 0, 0]
            rank_merit_image = preprocess_image(image, invert=False, scale=4, padsize=10, duplications=0)
            # small erosion
            structuring_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            #structuring_element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
            rank_merit_image = cv2.erode(rank_merit_image, structuring_element, iterations=1)

            rank_merit_image = cv2.GaussianBlur(rank_merit_image, (0, 0), 3)
            #imshow(rank_merit_image, cmap="gray")
            rank_merit_all = self._readtext(rank_merit_image, mode="line", engine=engine, allowed_chars="0123456789/")

            rank_merit = rank_merit_all[:rank_merit_all.find("/")]
            rank_merit = ''.join([i for i in rank_merit if i.isdigit()])
            if len(rank_merit) == 0:
                rank_merit = 0
            else:
                rank_merit = int(rank_merit)

            if self._record:
                return rank_merit, rank_merit_image, rank_merit_all
            else:
                return rank_merit

def has_two_lines_alliance_removal(image):

    max_lines = image.min(axis=1)
    max_lines = max_lines > 140
    max_lines = max_lines.astype(np.uint8)
    ccs = cv2.connectedComponentsWithStats(max_lines)
    if ccs[0] < 3:
        return False
    #height between lines
    ccs_y = ccs[3][:, 1]
    heights = ccs_y[1:] - ccs_y[:-1]
    count = np.count_nonzero(heights > 15)

    return count >= 2  # background, bottom line, top line

_par_left_one_line = imageio.imread('screenshots/par_left2.png').mean(axis=2).astype(np.uint8)
_par_right_one_line = imageio.imread('screenshots/par_right2.png').mean(axis=2).astype(np.uint8)

_par_left_two_lines = imageio.imread('screenshots/par_left.png').mean(axis=2).astype(np.uint8)
_par_right_two_lines = imageio.imread('screenshots/par_right.png').mean(axis=2).astype(np.uint8)

def detect_alliance_box(image, two_lines=False):
    #image = 255 - image
    par_left = _par_left_two_lines if two_lines else _par_left_one_line
    par_right = _par_right_two_lines if two_lines else _par_right_one_line
    matchleft = imgu.find_leftest_pattern(image, par_left)
    if matchleft is None:
        return None
    matchright = imgu.find_leftest_pattern(image, par_right)
    if matchright is None:
        return None
    outer_box = (matchleft[0], matchleft[1], matchright[0] + par_right.shape[1], matchright[1] + par_right.shape[0])
    inner_box = (matchleft[0] + par_left.shape[1], matchleft[1], matchright[0] + 1, matchright[1] + par_right.shape[0])
    return {'outer': outer_box, 'inner': inner_box}

class ReaderAllianceName(Reader):

    def __init__(self):
        super().__init__(type(self).__name__)
        self.textReader = ReaderText()

    def _read(self, image, tag, **kwargs):
        engine = kwargs.get('engine', 'easyocr')

        image = preprocess_image(image, invert=False, scale=2, padsize=0, duplications=0)
        #imshow(image, cmap="gray")
        two_lines = has_two_lines_alliance_removal(image)
        boxes = detect_alliance_box(image, two_lines=two_lines)

        if boxes is None and two_lines is False:
            boxes = detect_alliance_box(image, two_lines=True)

        # box should be on the left and not too far on the right
        if boxes is None or boxes['inner'][0] > 20 or (boxes['inner'][2] - boxes['inner'][0]) > 150:
            return self.try_parse_name(image, engine=engine, tag=tag)

        # if box inner left is too far on the right, it is not an alliance box

        alliance_box = boxes['inner']

        alliance_image = image[alliance_box[1]:alliance_box[3], alliance_box[0]:(alliance_box[2] - 2)]
        # alliance_image = np.pad(alliance_image, ((5, 5), (5, 5)), 'constant', constant_values=255)
        alliance = None
        if alliance_image.shape[0] < 10 or alliance_image.shape[1] < 10:
            alliance = ""
        else:
            #alliance_image = pad_image(alliance_image, padsize=10)
            # imshow(alliance_image, cmap="gray")
            duplications = 0 if alliance_image.shape[1] > 60 else 3
            if duplications > 0:
                try:
                    alliance = self.textReader.read(alliance_image, tag=tag, mode="line", engine=engine,
                                                    duplications=duplications, invert=True, scale=1)
                except AOOErrorOCR:
                    duplications = 0
            if duplications == 0:
                alliance = self.textReader.read(alliance_image, tag=tag, mode="line", engine=engine,
                                                duplications=duplications, invert=True, scale=1)

        name_image = self.trim_image_alliance_removal(
            self.remove_alliance_name_image(image, boxes, two_lines=two_lines))
        # imshow(name_image, cmap="gray")
        name = self.textReader.read(name_image, tag=tag, mode="line", engine=engine, invert=True)
        if self._record:
            return (alliance, name), None, None
        else:
            return alliance, name

    def try_parse_name(self, image, tag, engine='easyocr'):
        text = self.textReader.read(image, engine=engine, tag=tag, invert=True, scale=1)
        # regex "(XYZ) Name" or "(XYZ_1) Name" or "(XYZ_1_2) Name"
        import re
        m = re.match(r"\((\w{3,7})\) (.+)", text)
        if m is not None:
            res =  m.group(1), m.group(2)
        else:
            res = "", text
        if self._record:
            return res, image, text
        else:
            return res

    def remove_alliance_name_image(self, image, box, two_lines=False):
        #box = detect_alliance_box(image, two_lines=two_lines)
        if box is None:
            return image
        image = image.copy()
        extra = 4 if two_lines else 5
        outer_box = box['outer']
        minx, miny = outer_box[:2]
        maxx, maxy = outer_box[2:]
        minx = max(0, minx - extra)
        miny = max(0, miny - extra)
        maxx = min(image.shape[1], maxx + extra)
        maxy = min(image.shape[0], maxy + extra)
        image[miny:maxy, minx:maxx] = 255
        return image

    def trim_image_alliance_removal(self, image):
        op = image.copy()
        # imshow(op, cmap="gray")
        op[op > 150] = 0  # 105
        # imshow(op, cmap="gray")
        coords = cv2.findNonZero(op)  # Find all non-zero points (text)
        x, y, w, h = cv2.boundingRect(coords)  # Find minimum spanning bounding box
        x = max(x - 5, 0)
        y = max(y - 5, 0)
        w = min(w + 10, image.shape[1] - x)
        h = min(h + 10, image.shape[0] - y)
        image = image[y:y + h, x:x + w]  # Crop the image - note we do this on the original image
        # imshow(image)
        return image

class ReaderLuxuriousness(Reader):

    def __init__(self):
        super().__init__(type(self).__name__)
        self.large_pattern = cv2.imread('patterns/Luxuriousness_Large.png')
        self.large_pattern_limit = 88
        self.small_pattern_limit = 81

    def _read(self, image, tag, **kwargs):

        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        correlation = cv2.matchTemplate(image_bgr, self.large_pattern, cv2.TM_CCOEFF_NORMED)
        corr_max = np.max(correlation)
        if corr_max > 0.75:
            score_image = image[:, self.large_pattern_limit:, :]
        else:
            score_image = image[:, self.small_pattern_limit:, :]
        #imshow(score_image, cmap="gray")
        score_image = preprocess_image(score_image, invert=True, scale=2, padsize=10, duplications=0)
        score = self._readtext(score_image, mode='digits', engine='easyocr', allowed_chars='0123456789')
        if len(score) == 0:
            score = 0
        else:
            score = int(score)

        if self._record:
            return score, score_image, score
        else:
            return score


readerLargeNumbers = ReaderLargeNumbers()
readerSmallNumbers = ReaderSmallNumbers()
readerRanks = ReaderRank()
readerAllianceName = ReaderAllianceName()
readerText = ReaderText()
readerMeritProgression = ReaderMeritProgression()
readerLuxuriousness = ReaderLuxuriousness()

def luxuriousnessTest():
    import pickle
    from utils import get_box
    base = "screenshots/Luxuriousness/"
    files = [10, 92, 727, 772, 3111, 5429, 17678, 24691]

    city_frame_outline = "city_info_frame.pickle"
    with open(city_frame_outline, 'rb') as f:
        data = pickle.load(f)
    city_info_luxuriousness_box = data["rectangles"]["luxuriousness"]

    ocrreader = ReaderLuxuriousness()
    for f in files:
        image = imageio.imread(base + str(f) + ".png")
        luxuriousness_image = get_box(image, city_info_luxuriousness_box)
        imageio.imwrite(base + str(f) + "_luxuriousness.png", luxuriousness_image)
        v = ocrreader.read(luxuriousness_image, tag=["test"])

        print(f, v, "KO" if v != f else "OK")


if __name__ == "__main__":
    luxuriousnessTest()
    #image = imageio.imread("failed_nation_number.png")
    #op = readerText.read(image, tag="test", mode="all", engine="easyocr", scale=2, duplications=0, threshold=60)