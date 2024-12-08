import pickle

import numpy as np
import cv2
import PIL
import random
import imageio
from typing import NamedTuple
from recordclass import RecordClass
from utils import *
from pywinauto import mouse, keyboard
import time
import pyperclip
from aooutils.ocr import readerLargeNumbers, readerSmallNumbers, readerText, readerMeritProgression, preprocess_image, readerLuxuriousness

#readerLargeNumbers = ReaderLargeNumbers("Large numbers")
#readerSmallNumbers = ReaderSmallNumbers("Small numbers")
#readerText = ReaderText("Text")
#readerMeritProgression = ReaderMeritProgression("Merit progression")
#readerLuxuriousness = ReaderLuxuriousness()

class CityStats(RecordClass):
    name: str
    battle_power: int
    reputation: int
    kills: int
    losses: int
    luxuriousness: int
    merit_value: int
    city_level: int
    name_image: np.ndarray


action_sleep_time = 2

image_outline = "alliance_members_frame.pickle"
with open(image_outline, 'rb') as f:
    data = pickle.load(f)
alliance_members_page_R4_number_box = data["rectangles"]["R4_number"]
alliance_members_page_R3_number_box = data["rectangles"]["R3_number"]
alliance_members_page_R2_number_box = data["rectangles"]["R2_number"]
alliance_members_page_R1_number_box = data["rectangles"]["R1_number"]
alliance_members_page_R4_icon_box = data["rectangles"]["R4_icon"]
alliance_members_page_R3_icon_box = data["rectangles"]["R3_icon"]
alliance_members_page_R2_icon_box = data["rectangles"]["R2_icon"]
alliance_members_page_R1_icon_box = data["rectangles"]["R1_icon"]
alliance_members_page_R1_leader_icon_box = data["rectangles"]["leader_icon"]
alliance_members_page_R1_arrow_box = data["rectangles"]["arrow"]
alliance_members_main_frame_box = data["rectangles"]["main_frame"]
alliance_members_page_leader_name_box = data["rectangles"]["leader_name"]

cell_outline = "alliance_members_frame_cell.pickle"
with open(cell_outline, 'rb') as f:
    data = pickle.load(f)
alliance_members_cell_name_box = data["rectangles"]["name"]
alliance_members_cell_icon_box = data["rectangles"]["icon"]
alliance_members_cell_sword_box = data["rectangles"]["sword"]

city_frame_outline = "city_info_frame.pickle"
with open(city_frame_outline, 'rb') as f:
    data = pickle.load(f)
city_info_battle_power_box = data["rectangles"]["battle_power"]
city_info_copy_name_box = data["rectangles"]["copy_name"]
city_info_reputation_box = data["rectangles"]["reputation"]
city_info_kills_box = data["rectangles"]["kills"]
city_info_losses_box = data["rectangles"]["losses"]
city_info_luxuriousness_box = data["rectangles"]["luxuriousness"]
city_info_merit_button_box = data["rectangles"]["merit_button"]
city_info_merit_rank_box = data["rectangles"]["merite_rank"]
city_info_city_level_box = data["rectangles"]["city_level"]
close_banner_button_box = data["rectangles"]["close_banner_button"]

city_frame_outline2 = "city_info_frame2.pickle"
with open(city_frame_outline2, 'rb') as f:
    data = pickle.load(f)
city_info_battle_power_box2 = data["rectangles"]["battle_power"]
city_info_copy_name_box2 = data["rectangles"]["copy_name"]
city_info_reputation_box2 = data["rectangles"]["reputation"]
city_info_kills_box2 = data["rectangles"]["kills"]
city_info_losses_box2 = data["rectangles"]["losses"]
city_info_luxuriousness_box2 = data["rectangles"]["luxuriousness"]
city_info_color_test = data["rectangles"]["test_color"]
test_color = np.array([97, 94, 87])


city_merit_frame_outline = "city_info_merite_frame.pickle"
with open(city_merit_frame_outline, 'rb') as f:
    data = pickle.load(f)
city_merit_merit_box = data["rectangles"]["merite"]


#alliance_member_page_image_file = r"screenshots\alliance_members_frame.png"
alliance_member_page_image_file = r"patterns\alliance_members_frame.png"
alliance_member_page_image = cv2.imread(alliance_member_page_image_file)
arrow_image = get_box(alliance_member_page_image, alliance_members_page_R1_arrow_box) # alliance_member_page_image[alliance_members_page_R1_arrow_box[0][1]:alliance_members_page_R1_arrow_box[1][1], alliance_members_page_R1_arrow_box[0][0]:alliance_members_page_R1_arrow_box[1][0],:]

#alliance_member_page_cell_image_file = r"screenshots\alliance_members_frame_cell.png"
alliance_member_page_cell_image_file = r"patterns\alliance_members_frame_cell.png"
alliance_member_page_cell_image = cv2.imread(alliance_member_page_cell_image_file)
alliance_member_page_cell_image_height, alliance_member_page_cell_image_width, _ = alliance_member_page_cell_image.shape
alliance_member_page_cross_image = get_box(alliance_member_page_cell_image, alliance_members_cell_sword_box)
#alliance_member_page_cross_image = cv2.cvtColor(alliance_member_page_cross_image, cv2.COLOR_RGB2BGR)

alliance_summary_outline = "alliance_summary_frame.pickle"
with open(alliance_summary_outline, 'rb') as f:
    data = pickle.load(f)
alliance_summary_alliance_members_button_box = data["rectangles"]["alliance_members_button"]

def get_cell_box_from_sword_position(sword_position):
    shiftx = max(0,sword_position[0] - alliance_members_cell_sword_box[0][0] * alliance_member_page_cell_image_width)
    shifty = sword_position[1] - alliance_members_cell_sword_box[0][1] * alliance_member_page_cell_image_height
    return ((int(shiftx), int(shifty)),
            (int(alliance_member_page_cell_image_width + shiftx), int(alliance_member_page_cell_image_height + shifty)))

merit_values = {
    "recruit": 0,
    "private": 300,
    "private first class": 1300,
    "sergeant": 4300,
    "master sergeant": 11300,
    "warrant officer": 25300,
    "second lieutenant": 51300,
    "first lieutenant": 91300,
    "captain": 147300,
    "major": 222300,
    "lieutenant colonel": 322300,
    "colonel": 452300,
    "brigadier": 617300,
    "general": 822300,
    "general of the army": 1072300
}

merit_rank_diff = {
    "recruit": 300,
    "private": 1000,
    "private first class": 3000,
    "sergeant": 7000,
    "master sergeant": 14000,
    "warrant officer": 26000,
    "second lieutenant": 40000,
    "first lieutenant": 56000,
    "captain": 75000,
    "major": 100000,
    "lieutenant colonel": 130000,
    "colonel": 165000,
    "brigadier": 205000,
    "general": 250000,
    "general of the army": 999999999
}


def process_city_frame(image, window=None, tries=10):

    try:
        city_level_image = get_box(image, city_info_city_level_box)[:, 2:]
        city_level = readerSmallNumbers.read(city_level_image, invert=True, tag=["City info", "city level"])
    except:
        if tries > 0:
            print("Closing Banner")
            mouse.click(coords=get_box_center(image, close_banner_button_box))
            time.sleep(0.25)
            image = window.capture_as_image()
            image = np.array(image)
            return process_city_frame(image, window, tries=tries - 1)
        else:
            imageio.imwrite("screenshots/city_level_error.png", image)
            raise Exception("Cannot parse city level")

    mouse.click(coords=get_box_center(image, city_info_copy_name_box))
    time.sleep(0.25)
    # read the content of clipboard
    name = None
    pyperclip_tries = 5
    while pyperclip_tries > 0:
        try:
            name = pyperclip.paste()
            pyperclip_tries = 0
        except Exception as e:
            print("Pyperclip exception, retrying")
            time.sleep(0.5)
            pyperclip_tries -= 1
            if pyperclip_tries == 0:
                raise Exception('Pyperclip windows exception', e)


    luxuriousness_image = get_box(image, city_info_luxuriousness_box)
    luxuriousness = readerLuxuriousness.read(luxuriousness_image, tag=["City info", "luxuriousness"])
    #luxuriousness = readerText.read(luxuriousness_image, invert=True, tag=["City info", "luxuriousness"])
    #luxuriousness = luxuriousness.replace("o", "0")
    #luxuriousness = luxuriousness.replace("O", "0")
    #luxuriousness = ''.join([i for i in luxuriousness if i.isdigit()])
    #if luxuriousness == "":
    #    luxuriousness = 0
    #else:
    #    luxuriousness = int(luxuriousness)

    def read_city_info(image, battle_power_box, reputation_box, kills_box, losses_box):
        battle_power_image = get_box(image, battle_power_box)
        battle_power = readerLargeNumbers.read(battle_power_image, tag=["City info", "battle power"])

        kills_image = get_box(image, kills_box)
        kills = readerLargeNumbers.read(kills_image, tag=["City info", "kills"])

        losses_image = get_box(image, losses_box)
        losses = readerLargeNumbers.read(losses_image, tag=["City info", "losses"])

        reputation_image = get_box(image, reputation_box)
        reputation = readerLargeNumbers.read(reputation_image, tag=["City info", "reputation"])

        return battle_power, reputation, kills, losses

    # test if the layout includes merit rank
    test_box = get_box(image, city_info_color_test)
    mean_color = np.mean(test_box, axis=(0, 1))
    if np.sum(np.abs(mean_color - test_color)) < 10:
        battle_power, reputation, kills, losses = (
            read_city_info(image, city_info_battle_power_box2, city_info_reputation_box2, city_info_kills_box2, city_info_losses_box2))

        return name, battle_power, reputation, kills, losses, luxuriousness, 0, city_level
    else:
        battle_power, reputation, kills, losses = (
            read_city_info(image, city_info_battle_power_box, city_info_reputation_box, city_info_kills_box, city_info_losses_box))

        merit_rank_image = get_box(image, city_info_merit_rank_box)
        merit_rank = readerText.read(merit_rank_image, invert=True, tag=["City info", "merit rank"])

        merit_value = merit_values[merit_rank.lower()]

        if window is not None:
            mouse.click(coords=get_box_center(image, city_info_merit_button_box))
            image = capture_window(window, initial_sleep=0.1)
            #time.sleep(0.1)

            #image = window.capture_as_image()
            #image = np.array(image)
            merit_progression_image = get_box(image, city_merit_merit_box)
            val = readerMeritProgression.read(merit_progression_image, tag=["City info", "merit progression"])
            #val = process_military_frame(image)
            #print(name, merit_rank.lower(), val)
            if val < merit_rank_diff[merit_rank.lower()]:
                merit_value += val
            else:
                imageio.imwrite("error_merit_rank.png", image)
                # open input box with tkinter
                print("Mertit rank too high " + str(val) +"enter merit value:")
                try:
                    input1 = input()
                    val = int(input1)
                    merit_value += val
                    window.set_focus()
                    time.sleep(1)
                except Exception as e:
                    raise Exception("Merit rank is too high " + str(val))


            keyboard.send_keys("{ESC}")
            time.sleep(0.1)


        return name, battle_power, reputation, kills, losses, luxuriousness, merit_value, city_level

frame_validation_test_color = np.array([179.5, 174.9, 159.9])
def city_frame_validation(image):
    luxuriousness_box = get_box(image, city_info_luxuriousness_box)
    #imshow(luxuriousness_box)
    mean_color = np.mean(luxuriousness_box, axis=(0, 1))
    return np.sum(np.abs(mean_color - frame_validation_test_color)) > 10

def process_members(window, number, records, images, nation_data,  image=None):

    found_names = set()

    def is_same_image(image1, image2):
        corr = np.max(cv2.matchTemplate(image1, image2, cv2.TM_CCOEFF_NORMED))
        return corr > 0.95

    def already_found(icon):
        for i in range(len(images)):
            #corr = np.max(cv2.matchTemplate(images[i], icon, cv2.TM_CCOEFF_NORMED))
            if is_same_image(images[i][0], icon): #corr > 0.95:
                return images[i][1]
        return None
    last_image = None
    while len(found_names) < number:
        if window is not None:
            image = capture_window(window)
            #image = window.capture_as_image()
            #image = np.array(image)
        #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        frame_y_min = alliance_members_main_frame_box[0][1] * image.shape[0]
        frame_y_max = alliance_members_main_frame_box[1][1] * image.shape[0]
        image_opencv = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        op = cv2.matchTemplate(image_opencv, alliance_member_page_cross_image, cv2.TM_CCOEFF_NORMED)

        threshold = 0.75
        #imshow(image)
        #imshow(op)
        op2 = (op >= threshold).astype(np.uint8)

        ccs = cv2.connectedComponentsWithStats(op2, 4, cv2.CV_32S)
        centroids = ccs[3][1:]
        end = False


        for i, (x, y) in enumerate(centroids):
            #if y < alliance_members_main_frame_box[1][1]:
            #    continue
            cell_box = get_cell_box_from_sword_position((x, y))

            if cell_box[1][1] > frame_y_max or cell_box[0][1] < frame_y_min:
                #imshow(image[cell_box[0][1]:cell_box[1][1], cell_box[0][0]:cell_box[1][0], :])
                continue

            cell = image[cell_box[0][1]:cell_box[1][1], cell_box[0][0]:cell_box[1][0], :]
            cell_icon =get_box(cell, alliance_members_cell_name_box)
            #imshow(cell)
            #imshow(cell_icon)
            cell_icon = preprocess_image(cell_icon, padsize=0)
            #imshow(cell)
            if i == (len(centroids) - 1):
                if last_image is not None:
                    #imshow(cell)
                    #imshow(first_image)
                    if is_same_image(last_image, cell):
                        end = True
                        break
                last_image = cell

            #try:
            #    cell_icon = preproc_text_image(cell_icon, padsize=0)
            #except Exception as e:
            #    imshow(cell)
            #    raise e
            prev_name = already_found(cell)
            if prev_name is not None:
                found_names.add(prev_name)
                continue

            if window is not None:
                mouse.click(coords=(cell_box[0][0] + 10, cell_box[0][1] + 10))
                #time.sleep(action_sleep_time)
                #image2 = window.capture_as_image()
                #image2 = np.array(image2)
                image2 = capture_window(window, initial_sleep=0.5, validation_function=city_frame_validation)
                name, battle_power, reputation, kills, losses, luxuriousness, merit_value, city_level = process_city_frame(image2, window)


                if name not in found_names:
                    images.append((cell, name))
                    found_names.add(name)
                    print("Found", name, "stats:", battle_power, reputation, kills, losses, luxuriousness, merit_value, city_level)
                    r = CityStats(name, battle_power, reputation, kills, losses, luxuriousness, merit_value, city_level, cell_icon)
                    records[name] = r
                    flag = True
                else:
                    print("Warning, already found", name)

                keyboard.send_keys("{ESC}")
                time.sleep(0.3)
        if len(found_names) % 10 == 0:
            nation_data.save()
        if end or window is None:
            break

        window.set_focus()
        # mouse.scroll(coords=(int(x), int(y)), wheel_dist=int(-1))
        mouse_scroll(window, -160, position=(int(x), int(y)))
        time.sleep(1.5)
        # if number - len(found_names) == 1: # last one, sleep during the swing effect
        #    time.sleep(2)
        #    cv2.rectangle(image, cell_box[0], cell_box[1], (0, 255, 0), 2)
        #    imshow(image)

    if len(found_names) != number:
        raise Exception("Not enough members found", len(found_names), number)
    return records


def process_alliance_members_page(window, nation_data, alliance_short_txt, image=None):
    key = "tmp_alliance_members_page_" + alliance_short_txt
    if key in nation_data.data:
        records, images = nation_data.data[key]
    else:
        records = {}
        images = []
        nation_data.data[key] = (records, images)

    if image is None:
        image = capture_window(window)
        #image = window.capture_as_image()
        #image = np.array(image)
    #r4_number_text = extract_text(preproc_text_image(get_box(image, alliance_members_page_R4_number_box), padsize=0), mode='line')
    r4_number_image = get_box(image, alliance_members_page_R4_number_box)
    r4_number_text = readerText.read(r4_number_image, allowed_chars="0123456789/", tag=["Alliance members", "num players"])
    r4_number = int(r4_number_text[:r4_number_text.find("/")])

    alliance_members_page_R3_icon = get_box(image, alliance_members_page_R3_icon_box)
    #r3_number_text = extract_text(preproc_text_image(get_box(image, alliance_members_page_R3_number_box), padsize=0), mode='line')
    r3_number_image = get_box(image, alliance_members_page_R3_number_box)
    r3_number_text = readerText.read(r3_number_image, allowed_chars="0123456789/", tag=["Alliance members", "num players"])
    r3_number = int(r3_number_text[:r3_number_text.find("/")])

    alliance_members_page_R2_icon = get_box(image, alliance_members_page_R2_icon_box)
    #r2_number_text = extract_text(preproc_text_image(get_box(image, alliance_members_page_R2_number_box), padsize=0), mode='line')
    r2_number_image = get_box(image, alliance_members_page_R2_number_box)
    r2_number_text = readerText.read(r2_number_image, allowed_chars="0123456789/", tag=["Alliance members", "num players"])
    r2_number = int(r2_number_text[:r2_number_text.find("/")])

    alliance_members_page_R1_icon = get_box(image, alliance_members_page_R1_icon_box)
    r1_number_image = get_box(image, alliance_members_page_R1_number_box)
    #r1_number_text = extract_text(preproc_text_image(r1_number_image[:,2:], padsize=0), mode='line')
    r1_number_text = readerText.read(r1_number_image, allowed_chars="0123456789/", tag=["Alliance members", "num players"])
    r1_number = int(r1_number_text)

    #cell_icon = preproc_text_image(get_box(image, alliance_members_page_leader_name_box), padsize=0)
    cell_icon_image = get_box(image, alliance_members_page_leader_name_box)
    cell_icon = preprocess_image(cell_icon_image, padsize=0)
    #imshow(cell_icon)

    mouse.click(coords=get_box_center(image, alliance_members_page_R1_leader_icon_box))
    #time.sleep(action_sleep_time)
    #image = window.capture_as_image()
    #image = np.array(image)
    image = capture_window(window, initial_sleep=0.5, validation_function=city_frame_validation)
    name, battle_power, reputation, kills, losses, luxuriousness, merit_value, city_level = process_city_frame(image, window)
    r = CityStats(name, battle_power, reputation, kills, losses, luxuriousness, merit_value, city_level, cell_icon)
    records[name] = r
    keyboard.send_keys("{ESC}")
    time.sleep(0.25)

    def go_up_to_r3():
        flag = True
        while flag:
            image = capture_window(window)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            vmax = np.max(cv2.matchTemplate(image, alliance_members_page_R3_icon, cv2.TM_CCOEFF_NORMED))
            if vmax > 0.9:
                flag = False
            else:
                window.set_focus()
                op = cv2.matchTemplate(image, alliance_member_page_cross_image, cv2.TM_CCOEFF_NORMED)
                threshold = 0.75
                op2 = (op >= threshold).astype(np.uint8)
                ccs = cv2.connectedComponentsWithStats(op2, 4, cv2.CV_32S)
                centroids = ccs[3][1:]

                x, y = centroids[1]
                mouse_scroll(window, 170, position=(int(x+10), int(y+10)))
                time.sleep(1.5)


    def back_to_start():
        keyboard.send_keys("{ESC}")
        time.sleep(0.3)
        mouse.click(coords=get_box_center(image, alliance_summary_alliance_members_button_box))
        time.sleep(action_sleep_time * 2.5)

    skeep_r4 = False
    if not skeep_r4:
        mouse.click(coords=get_box_center(image, alliance_members_page_R4_icon_box))
        print("Extracting R4 infos...")
        time.sleep(action_sleep_time)
        process_members(window, r4_number, records, images, nation_data)
        nation_data.save()
        #mouse.scroll(coords=center(window), wheel_dist=int(-1))
        time.sleep(action_sleep_time)


    skeep_r3 = False
    if not skeep_r3:
        back_to_start()
        image = window.capture_as_image()
        image = np.array(image)
        pos = cv2.minMaxLoc(cv2.matchTemplate(image, alliance_members_page_R3_icon, cv2.TM_CCOEFF_NORMED))[3]
        mouse.click(coords=(pos[0] + 10, pos[1] + 10))
        print("Extracting R3 infos...")
        time.sleep(action_sleep_time)
        #go_up_to_r3()
        process_members(window, r3_number, records, images, nation_data)
        nation_data.save()
    #mouse.scroll(coords=center(window), wheel_dist=int(-1))

    skeep_r2 = False
    if not skeep_r2:

        time.sleep(action_sleep_time)
        back_to_start()
        image = window.capture_as_image()
        image = np.array(image)
        pos = cv2.minMaxLoc(cv2.matchTemplate(image, alliance_members_page_R2_icon, cv2.TM_CCOEFF_NORMED))[3]
        mouse.click(coords=(pos[0] + 10, pos[1] + 10))
        print("Extracting R2 infos...")
        time.sleep(action_sleep_time)
        #go_up_to_r3()
        process_members(window, r2_number, records, images, nation_data)
        nation_data.save()

    #mouse.scroll(coords=center(window), wheel_dist=int(-1))
    time.sleep(0.25)
    back_to_start()
    image = window.capture_as_image()
    image = np.array(image)
    pos = cv2.minMaxLoc(cv2.matchTemplate(image, alliance_members_page_R1_icon, cv2.TM_CCOEFF_NORMED))[3]
    mouse.click(coords=(pos[0] + 10, pos[1] + 10))
    print("Extracting R1 infos...")
    time.sleep(action_sleep_time)
    #go_up_to_r3()
    process_members(window, r1_number, records, images, nation_data)
    nation_data.save()
    #print(records)
    return records

if __name__ == "__main__":
    #im = cv2.imread(r"screenshots\alliance_members_frame_opened.png")
    #process_alliance_members_page_image(im)
    #process_members(im, 10)

    #im = imageio.imread(r"screenshots\commander_info_frame_prob_city_level.png")
    #print(process_city_frame(im, window=None))

    #im = imageio.imread(r"screenshots\error_city_info_frame4_large_name.png")

    #process_city_frame(im)
    #im = imageio.imread(r"screenshots\alliance_members_frame_large_name2.png")
    #print(process_members(None, 99, image=im))

    #process_military_frame(im)
    #process_alliance_members_page(None, im)
    #file = "nation_ranking_13-07-2023_09h-17m-07s.pickle"
    #with open(file, "rb") as f:
    #    records = pickle.load(f)
    #print(records)
    truth = [0, 22039, 73839, 50689, 42959, 782, 52448, 56544, 14427, 67132]
    flag = True
    for i in range(1, 10):
        im = imageio.imread(r"screenshots\error_city_info_merit_frame" + str(i) + ".png")
        if im.shape[2] == 4:
            im = im[:, :, :3]
        merit_progression_image = get_box(im, city_merit_merit_box)
        imshow(merit_progression_image)
        #process_members(None, 100, {}, [], None, image=im)
        val = readerMeritProgression.read(merit_progression_image, tag=["City info", "merit progression"])
        if truth[i] != val:
            flag = False
            print("Error", i, val)
    if flag:
        print("All good")