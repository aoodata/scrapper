import pickle

import numpy as np
import cv2
import PIL
import random
import imageio

from recordclass import RecordClass
from pywinauto import mouse, keyboard
from enum import Enum
from processAllianceMembersPage import process_alliance_members_page
import time
from utils import *

import aooutils.image as imgu
from aooutils.ocr import readerLargeNumbers, readerSmallNumbers, readerRanks, readerAllianceName, readerText, preprocess_image
action_sleep_time = 2

#readerLargeNumbers = ReaderLargeNumbers("Large numbers")
#readerSmallNumbers = ReaderSmallNumbers("Small numbers")
#readerRanks = ReaderRank("Ranks")
#readerAllianceName = ReaderAllianceName("Alliance and commander name")

class RankType(Enum):
    DEFAULT = 0
    ISLAND = 1
    MERIT = 2
    CROSS_NATION_ALLIANCE = 3
    CROSS_NATION_COMMANDER = 4
    ALLIANCE_ELITE = 5
    ALLIANCE_TERRITORY = 6
    COMMANDER_LEVEL = 7
    COMMANDER_CITY = 8
    ALLIANCE_MEMBERS = 9
    ALLIANCE_DEFAULT = 10

#(rank_txt, name_txt, alliance_short_txt, value_txt, name)
class RankingRecord(RecordClass):
    rank: int
    name: str
    alliance_short: str
    score: int
    name_image: np.ndarray
    nation: int
    image_version: int  # 1 with alliance name, 2 without alliance name (1 should never be stored in DB)
    original_name_image: np.ndarray = None


#from tesseractAPI import tessAPI, tessAPILine, RIL

#first_rank_mean = np.array([185.58750,153.07250,79.41250])
#second_rank_mean = np.array([150.15250,170.73250,166.80000])
#third_rank_mean = np.array([145.25500,106.68000,86.75000])






image_outline = "ranking_frame.pickle"
with open(image_outline, 'rb') as f:
    data = pickle.load(f)
ranking_box = data["rectangles"]["ranking"]

cell_outline = "ranking_cell.pickle"
with open(cell_outline, 'rb') as f:
    data = pickle.load(f)
cell_rect = data["rectangles"]

cell_outline = "merit_ranking_cell.pickle"
with open(cell_outline, 'rb') as f:
    data = pickle.load(f)
merit_cell_rect = data["rectangles"]

cell_outline = "city_ranking_cell.pickle"
with open(cell_outline, 'rb') as f:
    data = pickle.load(f)
city_cell_rect = data["rectangles"]

cell_outline = "island_ranking_cell.pickle"
with open(cell_outline, 'rb') as f:
    data = pickle.load(f)
island_cell_rect = data["rectangles"]

cell_outline = "alliance_ranking_cell.pickle"
with open(cell_outline, 'rb') as f:
    data = pickle.load(f)
alliance_cell_rect = data["rectangles"]

cell_outline = "alliance_elite_ranking_cell.pickle"
with open(cell_outline, 'rb') as f:
    data = pickle.load(f)
alliance_elite_cell_rect = data["rectangles"]

cell_outline = "alliance_territory_ranking_cell.pickle"
with open(cell_outline, 'rb') as f:
    data = pickle.load(f)
alliance_territory_cell_rect = data["rectangles"]

cell_outline = "cross_nation_alliance_ranking_cell.pickle"
with open(cell_outline, 'rb') as f:
    data = pickle.load(f)
cn_alliance_cell_rect = data["rectangles"]

cell_outline = "cross_nation_commander_ranking_cell.pickle"
with open(cell_outline, 'rb') as f:
    data = pickle.load(f)
cn_commander_cell_rect = data["rectangles"]

alliance_summary_outline = "alliance_summary_frame.pickle"
with open(alliance_summary_outline, 'rb') as f:
    data = pickle.load(f)
alliance_summary_alliance_members_button_box = data["rectangles"]["alliance_members_button"]





def process_ranking_screenshot(image,maxRank, ranktype, nation_data, key, save_cell_image=False, window=None, filter=None):

    ranking_image_y = image.shape[0] * ranking_box[0][1]
    ranking_image = get_box(image, ranking_box)
    #imshow(ranking_image)
    ###############################
    # Segment cells
    ###############################
    right_border = ranking_image[:, -3:, :]
    #imshow(right_border)
    right_border = cv2.cvtColor(right_border, cv2.COLOR_BGR2GRAY)
    right_border = np.mean(right_border, axis=1)

    brightness_threshold = 30
    cell_min_height = 70
    right_border_thr = right_border > brightness_threshold
    #imshow(right_border[:,None], cmap="gray")
    ccs = cv2.connectedComponentsWithStats(right_border_thr.astype(np.uint8), 4, cv2.CV_32S)
    centroids = ccs[3][1:, 1]

    cells = []
    prev = 0
    for i in range(len(centroids)):
        cur = int(centroids[i])
        if cur - prev > cell_min_height:
            if cur - prev > 90:
                cells.append((prev, prev + 85))
                cells.append((prev + 85, cur))
            else:
                cells.append((prev, cur))

        prev = cur
    cur = ranking_image.shape[0]
    if cur - prev > cell_min_height:
        cells.append((prev, cur))

    ###############################
    # Segment cells
    ###############################
    if ranktype == RankType.MERIT:
        rect = merit_cell_rect
        value_reader = readerLargeNumbers
    elif ranktype == RankType.CROSS_NATION_ALLIANCE:
        rect = cn_alliance_cell_rect
        value_reader = readerLargeNumbers
    elif ranktype == RankType.CROSS_NATION_COMMANDER:
        rect = cn_commander_cell_rect
        value_reader = readerLargeNumbers
    elif ranktype == RankType.ALLIANCE_ELITE:
        rect = alliance_elite_cell_rect
        value_reader = readerLargeNumbers
    elif ranktype == RankType.ALLIANCE_TERRITORY:
        rect = alliance_territory_cell_rect
        value_reader = readerLargeNumbers
    elif ranktype == RankType.COMMANDER_LEVEL or ranktype == RankType.COMMANDER_CITY:
        rect = city_cell_rect
        value_reader = readerSmallNumbers
    elif ranktype == RankType.ALLIANCE_MEMBERS or ranktype == RankType.ALLIANCE_DEFAULT:
        rect = alliance_cell_rect
        value_reader = readerLargeNumbers
    elif ranktype == RankType.ISLAND:
        rect = island_cell_rect
        value_reader = readerLargeNumbers
    else:
        rect = cell_rect
        value_reader = readerLargeNumbers

    rank_box = rect["rank"]
    #avatar_box = rect["avatar"]
    name_box = rect["name"]
    value_box = rect["value"]
    #alliance_box = rect["alliance"]

    if ranktype == RankType.ALLIANCE_MEMBERS:
        entries = {}
    else:
        entries = []
    prevRank = None
    for cell in cells:
        sub = ranking_image[cell[0]:cell[1], :, :]

        if save_cell_image:
            imageio.imwrite("screenshots/cell_image.png", sub)
            imshow(sub)
            save_cell_image = False


        try:

            #avatar = get_box(sub, avatar_box)


            name = get_box(sub, name_box)[:, 0:-5] #preproc_text_image(get_box(sub, name_box)[:, 0:-5], padsize=0)
            #name_preproc = preproc_text_image(get_box(sub, name_box)[:, 0:-5])
            name_image = preprocess_image(name, padsize=0)
            #imshow(name_image)

            #alliance_short_txt, name_txt = parse_name_image(name)
            alliance_short_txt, name_txt = readerAllianceName.read(name, tag=[str(ranktype), "alliance_and_name"])
            alliance_short_txt = alliance_short_txt.strip().lower()
            #print(alliance_short_txt, name_txt)
            name_txt = name_txt.strip()
            if name_txt == "":
                name_txt = "Unknown_" + str(random.randint(100000000, 1000000000))

            rank_image = get_box(sub, rank_box)
            #imshow(rank_image)
            rank = readerRanks.read(rank_image, tag=[str(ranktype), "rank"])

            value_image = get_box(sub, value_box)
            value = value_reader.read(value_image, tag=[str(ranktype), "score"])

            if ranktype == RankType.ALLIANCE_MEMBERS:
                if alliance_short_txt == "":
                    print("Error empty alliance short name", name_txt)
                    #imageio.imsave("error_alliance.png", name_preproc)
                    raise Exception("Error empty alliance short name")

                if filter is not None and alliance_short_txt.lower() not in filter:
                    print("Skipping alliance detailed members:", alliance_short_txt)
                    continue

                if key + "tmp" not in nation_data.data:
                    nation_data.data[key + "tmp"] = {}
                if alliance_short_txt not in nation_data.data[key + "tmp"]:
                    mouse.click(coords=(20, int(ranking_image_y + cell[0]) + 20))
                    time.sleep(action_sleep_time)

                    mouse.click(coords=get_box_center(image, alliance_summary_alliance_members_button_box))
                    time.sleep(action_sleep_time * 2.5)
                    try:
                        op = process_alliance_members_page(window, nation_data, alliance_short_txt)
                    except Exception as e:
                        print("Error processing alliance members page", e)
                        raise Exception("Error processing alliance members page", e)
                    nation_data.data[key + "tmp"][alliance_short_txt] = op
                    nation_data.save()
                    keyboard.send_keys("{ESC}")
                    time.sleep(0.5)
                    keyboard.send_keys("{ESC}")
                    time.sleep(0.5)
                else:
                    op = nation_data.data[key + "tmp"][alliance_short_txt]

                entries[alliance_short_txt] = op


            else:
                nation = 0
                if ranktype == RankType.CROSS_NATION_ALLIANCE or ranktype == RankType.CROSS_NATION_COMMANDER:
                    #nation = extract_text(preproc_text_image(get_box(sub, rect["nation"])))
                    nation = readerText.read(get_box(sub, rect["nation"]), tag=[str(ranktype), "nation"])
                    nstart = nation.find("#")
                    if nstart != -1:
                        nation = nation[nstart + 1:]
                        nation = ''.join([i for i in nation if i.isdigit()])
                        nation = int(nation)
                    else:
                        #if alliance_short_txt == "KSK":
                        #    nation = 460
                        #else:
                        imshow(sub)
                        raise Exception("Cannot parse nation")

                #entries.append((rank_txt, name_txt, alliance_short_txt, value_txt, name))
                entries.append(RankingRecord(rank, name_txt, alliance_short_txt, value, name_image, nation, 1))
            print(alliance_short_txt, name_txt, value, rank)
            if rank >= maxRank:
                break
        except Exception as e:
            print("Error processing cell", e)
            imageio.imwrite("error_cell.png", sub)
            imageio.imwrite("error_ranking.png", image)
            raise Exception(e)

    return entries



def debug(image_file, ranktype=RankType.DEFAULT):
    image = imageio.imread(image_file)
    data = process_ranking_screenshot(image, ranktype=ranktype, save_cell_image=True, maxRank=100, nation_data={}, key="dummy")
    #print(data)



#def remove_alliance_name(image):
#    tessAPILine.SetImage(PIL.Image.fromarray(image))
#    boxes = tessAPILine.GetComponentImages(RIL.WORD, True)
#    if len(boxes) <= 1:
#        return image
#    alliance_name = boxes[0][1]
#    image[alliance_name['y']-2:alliance_name['y'] + alliance_name['h']+4, alliance_name['x']-2:alliance_name['x'] + alliance_name['w']+4] = 255
#    return image

def proc_cell():
    image_file = "screenshots/ranking_cell.png"
    image = imageio.imread(image_file)
    name_box = cell_rect["name"]
    name = preproc_text_image(get_box(image, name_box))
    imshow(name)
    #name = remove_alliance_name(name)
    #imshow(name)

if __name__ == "__main__":
    #proc_cell()
    #debug("screenshots/alliance_power_ranking.png", ranktype=RankType.ALLIANCE_DEFAULT)
    debug("screenshots/cross_nation_alliance_ranking_problem.png", ranktype=RankType.CROSS_NATION_ALLIANCE)
    #debug("screenshots/ranking_city_frame.png", ranktype=RankType.COMMANDER_CITY)

#image_file = "screenshots/alliance_territory_ranking_frame.png"
#image = imageio.imread(image_file)
#data = process_ranking_screenshot(image, save_cell_image=True)
#image = cv2.imread(image_file)
#data = process_ranking_screenshot(image)
#print(data)
#print(data)

#image_file = "33.png"
#image = cv2.imread(image_file)
#txt = extract_text(image, mode="word")
#print(txt)