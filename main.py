bluestack_path = r'C:\Program Files\BlueStacks_nxt\HD-Player.exe'

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from datetime import datetime
from pywinauto import Application
from tkinter import *
from enum import Enum
from processRankingPage import RankType
from utils import *
import time
import imageio
import pprint


from aooutils.ocr import readerLargeNumbers, readerText

pp = pprint.PrettyPrinter(indent=4)

action_sleep_time = 2




def check_record_order(records, increasing):

    prev_score = float("inf") if not increasing else -float("inf")
    for record in records:
        if increasing:
            if record.score < prev_score:
                print("Warning: records not in order", record.name, record.score, prev_score)
                return False
        else:
            if record.score > prev_score:
                print("Warning: records not in order", record.name, record.score, prev_score)
                return False
        prev_score = record.score
    return True

def extract_ranking_info(window, max_rank, ranktype, nation_data, key, increasing=False, filter=None):


    from processRankingPage import process_ranking_screenshot
    # assume that a ranking page is open and scroll position is at the top
    data = {}
    screen_number = 0
    max_rank_found = 0
    while len(data) < max_rank:
        image = window.capture_as_image()
        image = np.array(image)

        screen_number += 1

        tries = 0
        max_tries = 7 if ranktype != RankType.ALLIANCE_MEMBERS else 1
        entries = None
        while True:
            try:
                # extract ranking info
                entries = process_ranking_screenshot(image, max_rank, ranktype, nation_data, key, window=window, filter=filter)
                if  isinstance(entries, list) and not check_record_order(entries, increasing):
                    for entry in entries:
                        print(entry.name, entry.rank, entry.score)
                    raise Exception("Records not in order")
                if ranktype != RankType.ALLIANCE_MEMBERS:
                    for e in entries:
                        if e.rank > max_rank_found + 1:
                            raise Exception("Missing record")
                        max_rank_found = max(max_rank_found, e.rank)
                break
            except Exception as e:
                tries += 1
                if tries >= max_tries:
                    raise Exception("Failed to extract ranking info", e)
                else:
                    print("Failed to extract ranking info, trying again")
                    if tries % 3 == 0:
                        print("Scrolling up")
                        window.set_focus()
                        mouse_scroll(window, 10)
                        time.sleep(2)
                    else:
                        time.sleep(0.25)

                    image = window.capture_as_image()
                    image = np.array(image)
        # extract ranking info
        #entries = process_ranking_screenshot(image, max_rank, ranktype, nation_data, key, window=window)

        flag = True
        if ranktype == RankType.ALLIANCE_MEMBERS:
            data.update(entries)
        else:
            # add to data
            prev_rank = -1
            for entry in entries:

                if entry.rank not in data:
                    flag = False
                    data[entry.rank] = entry


        if flag:
            break   # no new entries found
        # scroll down
        window.set_focus()
        mouse_scroll(window, -320)
        time.sleep(1)

    if (filter is None and len(data) < max_rank) or (filter is not None and len(data) < len(filter)):
        print("Warning: not enough data found " + str(len(data)) + "/" + str(max_rank))
        for i in range(1, max_rank+1):
            if i not in data:
                print("Missing rank: " + str(i))

        raise Exception("Not enough data found")

    nation_data.data[key] = data
    nation_data.save()
    return data

from aooutils.Navigator import Navigator

import cv2

#nation_ranking_images = {
#    "commander_power": cv2.imread("screenshots/nation_ranking_commander_power.png"),
#    "commander_kill": cv2.imread("screenshots/nation_ranking_commander_kill.png"),
#    "commander_city": cv2.imread("screenshots/nation_ranking_commander_city.png"),
#    "commander_officer": cv2.imread("screenshots/nation_ranking_commander_officer.png"),
#    "commander_titan": cv2.imread("screenshots/nation_ranking_commander_titan.png"),
#    "commander_warplane": cv2.imread("screenshots/nation_ranking_commander_warplane.png"),
#    "commander_island": cv2.imread("screenshots/nation_ranking_commander_island.png"),
#    "commander_merit": cv2.imread("screenshots/nation_ranking_commander_merit.png"),
#    "commander_level": cv2.imread("screenshots/nation_ranking_commander_level.png"),
#    "alliance_members": cv2.imread("screenshots/nation_ranking_alliance_power.png"),
#    "alliance_power": cv2.imread("screenshots/nation_ranking_alliance_power.png"),
#    "alliance_kill": cv2.imread("screenshots/nation_ranking_alliance_kill.png"),
#    "alliance_territory": cv2.imread("screenshots/nation_ranking_alliance_territory.png"),
#    "alliance_elite": cv2.imread("screenshots/nation_ranking_alliance_elite.png"),
#    #"alliance_mines": cv2.imread("screenshots/nation_ranking_alliance_mines.png"),
#    #"alliance_benefits": cv2.imread("screenshots/nation_ranking_alliance_benefits.png"),
#}

#nation_ranking_images = {
#    "commander_power": cv2.imread("patterns/nation_ranking_commander_power.png"),
#    "commander_kill": cv2.imread("patterns/nation_ranking_commander_kill.png"),
#    "commander_city": cv2.imread("patterns/nation_ranking_commander_city.png"),
#    "commander_officer": cv2.imread("patterns/nation_ranking_commander_officer.png"),
#    "commander_titan": cv2.imread("patterns/nation_ranking_commander_titan.png"),
#    "commander_warplane": cv2.imread("patterns/nation_ranking_commander_warplane.png"),
#    "commander_island": cv2.imread("patterns/nation_ranking_commander_island.png"),
#    "commander_merit": cv2.imread("patterns/nation_ranking_commander_merit.png"),
#    "commander_level": cv2.imread("patterns/nation_ranking_commander_level.png"),
#    "alliance_members": cv2.imread("patterns/nation_ranking_alliance_power.png"),
#    "alliance_power": cv2.imread("patterns/nation_ranking_alliance_power.png"),
#    "alliance_kill": cv2.imread("patterns/nation_ranking_alliance_kill.png"),
#    "alliance_territory": cv2.imread("patterns/nation_ranking_alliance_territory.png"),
#    "alliance_elite": cv2.imread("patterns/nation_ranking_alliance_elite.png"),
#}


# enum for detail level
class ScanLevel(Enum):
    FAST = 0
    NORMAL = 10
    DETAILED = 20

from recordclass import RecordClass
class RankingInfo(RecordClass):
    name: str
    type: RankType
    max_rank: int
    image: np.ndarray
    increasing: bool
    scan_level: ScanLevel
    filter: set = None

ranking_info = {
    "commander_power": RankingInfo("commander_power", RankType.DEFAULT, 100, cv2.imread("patterns/nation_ranking_commander_power.png"), False, ScanLevel.FAST),
    "commander_kill": RankingInfo("commander_kill", RankType.DEFAULT, 100, cv2.imread("patterns/nation_ranking_commander_kill.png"), False, ScanLevel.NORMAL),
    "commander_city": RankingInfo("commander_city", RankType.COMMANDER_CITY, 100, cv2.imread("patterns/nation_ranking_commander_city.png"), False, ScanLevel.FAST),
    "commander_officer": RankingInfo("commander_officer", RankType.DEFAULT, 100, cv2.imread("patterns/nation_ranking_commander_officer.png"), False, ScanLevel.FAST),
    "commander_titan": RankingInfo("commander_titan", RankType.DEFAULT, 100, cv2.imread("patterns/nation_ranking_commander_titan.png"), False, ScanLevel.FAST),
    "commander_warplane": RankingInfo("commander_warplane", RankType.DEFAULT, 100, cv2.imread("patterns/nation_ranking_commander_warplane.png"), False, ScanLevel.FAST),
    "commander_island": RankingInfo("commander_island", RankType.ISLAND, 100, cv2.imread("patterns/nation_ranking_commander_island.png"), False, ScanLevel.NORMAL),
    "commander_merit": RankingInfo("commander_merit", RankType.MERIT, 100, cv2.imread("patterns/nation_ranking_commander_merit.png"), False, ScanLevel.NORMAL),
    "commander_level": RankingInfo("commander_level", RankType.COMMANDER_LEVEL, 100, cv2.imread("patterns/nation_ranking_commander_level.png"), False, ScanLevel.NORMAL),
    "alliance_members": RankingInfo("alliance_members", RankType.ALLIANCE_MEMBERS, 3, cv2.imread("patterns/nation_ranking_alliance_power.png"), False, ScanLevel.DETAILED),
    "alliance_power": RankingInfo("alliance_power", RankType.ALLIANCE_DEFAULT, 10, cv2.imread("patterns/nation_ranking_alliance_power.png"), False, ScanLevel.FAST),
    "alliance_kill": RankingInfo("alliance_kill", RankType.ALLIANCE_DEFAULT, 10, cv2.imread("patterns/nation_ranking_alliance_kill.png"), False, ScanLevel.NORMAL),
    "alliance_territory": RankingInfo("alliance_territory", RankType.ALLIANCE_TERRITORY, 10, cv2.imread("patterns/nation_ranking_alliance_territory.png"), False, ScanLevel.NORMAL),
    "alliance_elite": RankingInfo("alliance_elite", RankType.ALLIANCE_ELITE, 10, cv2.imread("patterns/nation_ranking_alliance_elite.png"), True, ScanLevel.NORMAL),
}




with open("keranking_frame.pickle", 'rb') as f:
    data = pickle.load(f)
strongest_commander_rect = data["rectangles"]

with open("ke_phase_ranking_frame.pickle", 'rb') as f:
    data = pickle.load(f)
ke_ranking_rect = data["rectangles"]


def data_to_json(data, nation):
    rankings = ["commander_power", "commander_city", "commander_officer", "commander_titan", "commander_warplane"]

    all_alliances = set()
    all_data = {}
    for rn in rankings:
        scores = []
        alliances = []
        for r, v in data[rn].items():
            alliance_short = v.alliance_short
            score = v.score
            scores.append(score)
            alliances.append(alliance_short)
            all_alliances.add(alliance_short)

        # create a dataframe
        # df = pd.DataFrame({"alliance": alliances, "score": scores})
        all_data[rn + "_alliances"] = alliances
        all_data[rn + "_score"] = scores

    alliance_power_score = []
    alliance_power_names = []
    for r, v in data["alliance_power"].items():
        alliance_short = v.alliance_short
        score = v.score
        alliance_power_score.append(score)
        alliance_power_names.append(alliance_short)
        all_alliances.add(alliance_short)

    all_data["alliance_power_score"] = alliance_power_score
    all_data["alliance_power_names"] = alliance_power_names

    all_data["alliances"] = list(all_alliances)
    all_data["nation"] = nation

    all_data["date"] = data["date"]
    filename = f"cmp/fast_scan/nation_data_{nation}.json"
    with open(filename, "w") as f:
        import json
        json.dump(all_data, f)
    print("Finished and saved to " + filename)
    return all_data

def proc_cross_nation_global_ranking_page(window, data, image, nation, event_type):
    nation_score_rect = strongest_commander_rect["NationScore"]
    nation_name_rect = strongest_commander_rect["NationName"]
    other_nation_score_rect = strongest_commander_rect["OtherNationScore"]
    other_nation_name_rect = strongest_commander_rect["OtherNationName"]


    def nation_number(image):
        #image_proc = preproc_text_image(image, padsize=0, trim=False)
        #nation_info = extract_text(image_proc)
        nation_info = readerText.read(image, scale=2, duplications=0, threshold=60, tag=["Strongest Commander", "nation number"])
        start = nation_info.find("#")
        end = nation_info.find(")")
        try:
            num = int(nation_info[start + 1:end])
        except:
            print("Failed to extract nation number")
            imshow(image, cmap="gray")
            imageio.imwrite("failed_nation_number.png", image)
            num = int(input("Enter nation number: "))

            window.set_focus()
        return num

    other_nation_name_image = get_box(image, other_nation_name_rect)
    other_nation_name = nation_number(other_nation_name_image)

    #other_nation_score = score_txt_to_int(extract_text(preproc_text_image(get_box(image, other_nation_score_rect), trim=False)))
    other_nation_score = readerLargeNumbers.read(get_box(image, other_nation_score_rect), threshold=60, tag=["Strongest Commander", "nation score"])


    nation_name_image = get_box(image, nation_name_rect)
    nation_name = nation_number(nation_name_image)

    #nation_score = score_txt_to_int(extract_text(preproc_text_image(get_box(image, nation_score_rect), trim=False)))
    nation_score = readerLargeNumbers.read(get_box(image, nation_score_rect), threshold=60, tag=["Strongest Commander", "nation score"])

    if nation_name == nation:
        data["nation_" + event_type + "_score"] = nation_score
        data["other_nation_" + event_type + "_score"] = other_nation_score
        data["other_nation_name"] = other_nation_name
    elif other_nation_name == nation:
        data["nation_" + event_type + "_score"] = other_nation_score
        data["other_nation_" + event_type + "_score"] = nation_score
        data["other_nation_name"] = nation_name
    else:
        print("Nation not found in ranking page")
        print("Nation name: " + str(nation_name))
        print("Other nation name: " + str(other_nation_name))
        print("Given Nation: " + str(nation))
        raise Exception("Nation not found in ranking page")

    print("Nation score: " + str(data["nation_" + event_type + "_score"]))
    print("Other nation score: " + str(data["other_nation_" + event_type + "_score"]))
    print("Other nation name: " + str(data["other_nation_name"]))



def extract_strongest_commander_ranking(window, nation=385):
    global nation_data
    # assume that a ranking page is open and scroll position is at the top
    data = nation_data.data
    #ke_ranking_button_rect = strongest_commander_rect["KERankingButton"]
    total_ranking_button_rect = ke_ranking_rect["total_ranking_button"]
    commander_phase_ranking_button_rect = ke_ranking_rect["commander_phase_ranking_button"]

    alliance_ranking_button_rect = strongest_commander_rect["AllianceRankingButton"]
    commander_ranking_button_rect = strongest_commander_rect["CommanderRankingButton"]

    title_rect = strongest_commander_rect["Title"]

    rankings = [
        "commander_ke_frenzy_score",
        #"alliance_ke_frenzy_score",
        "commander_sc_frenzy_score",
        "alliance_sc_frenzy_score",
        "commander_ke_void_score",
        #"alliance_ke_void_score",
        "commander_sc_void_score",
        "alliance_sc_void_score",
    ]

    image = window.capture_as_image()
    image = np.array(image)

    #image = imageio.imread("screenshots/StrongestCommanderRanking.png")

    #title = extract_text(preproc_text_image(get_box(image, title_rect)))
    title = readerText.read(get_box(image, title_rect), tag=["Strongest Commander", "type"])
    if "void" in title.lower():
        event_type = "void"
    else:
        event_type = "frenzy"
    data["event_type"] = event_type

    def proc_ke_page(event_type):
        mouse.click(coords=get_box_center(image, commander_phase_ranking_button_rect))
        print("Extracting KE commander ranking")
        time.sleep(action_sleep_time)
        if "commander_" + event_type not in data:
            extract_ranking_info(window, 100, RankType.CROSS_NATION_COMMANDER, nation_data, "commander_" + event_type)
            nation_data.save()
        keyboard.send_keys("{ESC}")
        time.sleep(action_sleep_time)

    proc_ke_page("ke_" + event_type)

    mouse.click(coords=get_box_center(image, total_ranking_button_rect))
    time.sleep(action_sleep_time)

    def proc_sc_page(event_type):
        image = window.capture_as_image()
        image = np.array(image)

        proc_cross_nation_global_ranking_page(window, data, image, nation, event_type)

        #data["other_nation_name"] = other_nation_name
        #data["nation_" + event_type + "_score"] = score_txt_to_int(extract_text(preproc_text_image(get_box(image, nation_score_rect))))
        #data["other_nation_" + event_type + "_score"] = score_txt_to_int(extract_text(preproc_text_image(get_box(image, other_nation_score_rect))))

        mouse.click(coords=get_box_center(image, commander_ranking_button_rect))
        time.sleep(action_sleep_time)
        if "commander_" + event_type not in data:
            extract_ranking_info(window, 100, RankType.CROSS_NATION_COMMANDER, nation_data, "commander_" + event_type)
            nation_data.save()
        keyboard.send_keys("{ESC}")
        time.sleep(action_sleep_time)

        mouse.click(coords=get_box_center(image, alliance_ranking_button_rect))
        time.sleep(action_sleep_time)
        if "alliance_" + event_type not in data:
            extract_ranking_info(window, 15, RankType.CROSS_NATION_ALLIANCE, nation_data, "alliance_" + event_type)
            nation_data.save()
        keyboard.send_keys("{ESC}")

    proc_sc_page("sc_" + event_type)

    date_string = datetime.utcnow().strftime("%d-%m-%Y_%Hh-%Mm-%Ss")
    data["date"] = time.time()
    filename = str(nation) + "_cross_nation_ranking_" + date_string + ".pickle"
    with open(filename, "wb") as f:
        pickle.dump(data, f)
    print("Finished and saved to " + filename)
    #pp.pprint(data)

    return data


class TempData:

    def __init__(self, filename="tmp.pickle"):
        self.filename = filename
        self.data = {}
        # if file exists, load it
        try:
            self.load()
        except:
            pass

    def save(self):
        with open(self.filename, "wb") as f:
            pickle.dump(self.data, f)

    def load(self):
        with open(self.filename, "rb") as f:
            self.data = pickle.load(f)

    def clear(self):
        self.data = {}
        # remove file if exists
        try:
            import os
            os.remove(self.filename)
        except:
            pass



nation_data = None

def navigate_nation_rankings(window, scan_level=ScanLevel.NORMAL,nation="385"):
    # assume that a ranking page is open and scroll position is at the top
    global nation_data


    c = 0
    for key, rinfo in ranking_info.items():
        ranking_image = rinfo.image
        ranktype = rinfo.type
        max_rank = rinfo.max_rank
        rscan_level = rinfo.scan_level
        rincreasing = rinfo.increasing
        filter = rinfo.filter

        if rscan_level.value <= scan_level.value:

            print("Extracting " + key)

            if key in nation_data.data:# or key != "alliance_members":
                print("Already extracted " + key)
            else:

                image = window.capture_as_image()
                image = np.array(image)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                res = cv2.matchTemplate(image, ranking_image, cv2.TM_CCOEFF)
                #imshow(res)
                loc = np.unravel_index(np.argmax(res), res.shape)
                # left click at location

                mouse.click(coords=(loc[1]+ranking_image.shape[1]//2, loc[0]+ranking_image.shape[0]//2))
                time.sleep(action_sleep_time)


                extract_ranking_info(window, max_rank, ranktype, nation_data, key, rincreasing, filter)
                keyboard.send_keys("{ESC}")

                time.sleep(action_sleep_time)

        if key != "alliance_members":
            c+=1
            if c%2==0:
                window.set_focus()
                mouse_scroll(window, -110)
                time.sleep(0.75)

    date_string = datetime.utcnow().strftime("%d-%m-%Y_%Hh-%Mm-%Ss")
    nation_data.data["date"] = time.time()
    if scan_level == ScanLevel.FAST:

        data_to_json(nation_data.data, nation)
    else:
        filename = str(nation) + "_nation_ranking_" + date_string + ".pickle"
        with open(filename, "wb") as f:
            pickle.dump(nation_data.data, f)
        print("Finished and saved to " + filename)
    return nation_data.data


def op_extract_ranking():
    from processRankingPage import process_ranking_screenshot
    # assume that a ranking page is open and scroll position is at the top
    data = {}
    screen_number = 0

    while len(data) < 100:
        # take screenshot
        image = imageio.imread("power_ranking_" + str(screen_number) + ".png")
        # PIL image to numpy array

        #imshow(image)

        screen_number += 1
        # extract ranking info
        entries = process_ranking_screenshot(image)
        flag = True
        # add to data
        for entry in entries:
            if entry.rank not in data:
                flag = False
                data[entry.rank] = entry

        if flag:
            break  # no new entries found


    return data

def op_pywinauto():
    from tkinter import filedialog


    #app = Application().start(r'"C:\Program Files\BlueStacks_nxt\HD-Player.exe" --instance Nougat64_6')
    app = Application().connect(path=bluestack_path)
    dlg = app.top_window()

    navigator = Navigator(dlg)
    op=app.windows()

    print(dlg.rectangle())
    #mouse.move(coords=(r.left+30, r.top+50))
    #mouse.click("left",coords=(r.left+30, r.top+50))
    #(L0, T0, R504, B927)
    #winsize =(536, 927) # deforme
    winsize = (504, 927) # non deforme
    # resize app window
    dlg.set_focus()
    dlg.move_window(x=0, y=0, width=winsize[0], height=winsize[1])

    #print("tatat")
    #win = app['UntitledNotepad']
    #win.type_keys('hello world')

    #start = time.time()
    #op = np.array(capture_window(dlg))
    #end = time.time()
    #print(end - start)


    # tkinter open window
    root = Tk()
    root.title("Gogogo")
    root.geometry("300x500")

    #center window
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry('{}x{}+{}+{}'.format(width, height, x, y))

    #create a frame to hold buttons
    frame = Frame(root)
    frame.pack()

    def populate_nation_data():
        global nation_data
        if nation_data is None:

            nation = int(text.get("1.0", END))
            nation_data = TempData(str(nation) + "_tmp_nation_data.pickle")


    def take_screenshot():
        image = dlg.capture_as_image()#.save('screenshot.png')
        #choose filename from tkinter
        filename = filedialog.asksaveasfilename(initialdir = "/",title = "Select file",filetypes = (("png files","*.png"),("all files","*.*")))
        image.save(filename)

    #create a button tot ake screenshot
    button = Button(frame, text="Take Screenshot", command=take_screenshot)
    button.pack(side=BOTTOM)

    #def reset_nation_data():
    #    global nation_data
    #    nation_data = None
    #
    #button = Button(frame, text="Reset Nation Data", command=reset_nation_data)
    #button.pack(side=BOTTOM)

    def extract_ranking():
        populate_nation_data()
        nation = int(text.get("1.0", END))
        data = extract_strongest_commander_ranking(dlg, nation=nation)
        #print(data)

    button = Button(frame, text="Strongest Commander", command=extract_ranking)
    button.pack(side=BOTTOM)


    def navigate_nation_ranking_fast():
        populate_nation_data()
        nation = int(text.get("1.0", END))
        navigate_nation_rankings(dlg, nation=nation, scan_level=ScanLevel.FAST)
        nation_data.clear()


    button = Button(frame, text="N. Rankings Fast", command=navigate_nation_ranking_fast)
    button.pack(side=BOTTOM)

    #def navigate_nation_ranking():
    #    populate_nation_data()
    #    nation = int(text.get("1.0", END))
    #    navigate_nation_rankings(dlg, nation=nation)
    #
    #button = Button(frame, text="N. Rankings", command=navigate_nation_ranking)
    #button.pack(side=BOTTOM)

    def navigate_nation_ranking_detailed():
        populate_nation_data()
        nation = int(text.get("1.0", END))
        try:
            numDetailed = int(textNumAlliances.get("1.0", END))
            filter = None
        except:
            numDetailed = 10
            names = textNumAlliances.get("1.0", END)[:-1].split(" ")
            names = [name.lower() for name in names]
            filter = set(names)

        ranking_info["alliance_members"].max_rank = numDetailed
        ranking_info["alliance_members"].filter = filter
        #Reader.start_record()
        navigate_nation_rankings(dlg, scan_level=ScanLevel.DETAILED, nation=nation)
        #Reader.stop_record()

    button = Button(frame, text="N. Rankings + Members", command=navigate_nation_ranking_detailed)
    button.pack(side=BOTTOM)

    #add label "nation"
    label = Label(root, text="Nation")
    label.pack()

    # add text field
    text = Text(root, height=1, width=30)
    text.pack()

    label = Label(root, text="Detailed alliances")
    label.pack()
    # add text field with default value "3"
    textNumAlliances = Text(root, height=1, width=30)
    textNumAlliances.insert(END, "4")
    textNumAlliances.pack()

    def test_scroll():
        distance = int(text.get("1.0", END))
        mouse_scroll(dlg, distance)

    #button = Button(frame, text="Test Scroll", command=test_scroll)
    #button.pack(side=BOTTOM)

    def navigate_to():
        target = text.get("1.0", END)
        navigator.navigate(target[:-1])

    #button = Button(frame, text="go to", command=navigate_to)
    #button.pack(side=BOTTOM)

    def relocate():
        navigator.relocate()

    #button = Button(frame, text="relocate", command=relocate)
    #button.pack(side=BOTTOM)

    root.mainloop()





def debug():
    image = imageio.imread("screenshots/strongest_commander_problem.png")
    proc_cross_nation_global_ranking_page(None, data, image, 385, "frenzy")



def main():
    op_pywinauto()
    #extract_strongest_commander_ranking(None)



if __name__ == "__main__":
    #debug()
    main()
    ...



#test_pywinauto()
#data = op_extract_ranking()

#pp.pprint(data)

