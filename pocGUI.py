
websitepath = "../aoodata.github.io/"
dbpath = websitepath + "data/"

import webview
import imageio.v3 as imageio
from io import BytesIO
from PIL import Image
import random
import pickle
from utils import imagetobase64, first
from insert_rankings_to_db import insert_rankings_to_db_algorithm
from win11toast import toast
import win32gui

from pywinauto.findwindows    import find_window


#file = "commander_ranking.pickle"
#with open(file, 'rb') as f:
#    data_ranking = pickle.load(f)
#
#
#file = "commander_fusion.pickle"
#with open(file, 'rb') as f:
#    data_fusion = pickle.load(f)

def ranking_record_to_dict(record):
    return {
        "name": record.name,
        "alliance_short": record.alliance_short,
        "score": record.score,
        "rank": record.rank,
        "name_image": imagetobase64(record.name_image)
    }


class CommanderFusionData:

    def __init__(self, data):
        # dict: commander_name_in_ranking_data -> (commander_ranking_data, commander_name_in_db, score_diff)
        self.matched_commanders_to_insert = data["matched_commanders_to_insert"]
        # set: commander_name_in_ranking_data
        self.commanders_to_insert = data["commanders_to_insert"]
        # dict: commander_name_in_ranking_data -> list [(score_diff, commander_name_in_db)]
        self.matching_stats = data["matching_stats"]
        # dict: commander_name_in_ranking_data -> (dict: ranking_name -> score)
        self.commander_to_insert_scores = data["commander_to_insert_scores"]
        # dict: commander_name_in_db -> (dict: ranking_name -> score)
        self.commander_db_scores = data["commander_db_scores"]
        #
        self.non_matched_commanders_db = {}

        for key, item  in data["non_matched_commanders_db"].items():
            self.non_matched_commanders_db[key] = {"name": item[1] + " (" + item[2] + ")", "id": item[0]}


        #for c in self.matched_commanders_to_insert:
        #    t = self.matched_commanders_to_insert[c]
        #    self.matched_commanders_to_insert[c] = (t[0], t[1])

    def get_data(self):
        return {
            "matched_commanders_to_insert": self.matched_commanders_to_insert,
            "commanders_to_insert": list(self.commanders_to_insert),
            "matching_stats": self.matching_stats,
            "commander_to_insert_scores": self.commander_to_insert_scores,
            "commander_db_scores": self.commander_db_scores,
            "non_matched_commanders_db": self.non_matched_commanders_db
        }

    def add_matched_commander(self, commander_name_in_ranking_data, commander_name_in_db, score_diff):
        self.matched_commanders_to_insert[commander_name_in_ranking_data] = (commander_name_in_db, score_diff)
        self.commanders_to_insert.remove(commander_name_in_ranking_data)

    def remove_matched_commander(self, commander_name_in_ranking_data):
        del self.matched_commanders_to_insert[commander_name_in_ranking_data]
        self.commanders_to_insert.add(commander_name_in_ranking_data)



class RankingData:
    def __init__(self, data):
        self.ranking = data["commander_ranking"]
        self.diagnostic = data["diagnostic"]

    def get_ranking(self, full=False):


        commanders = None
        if full:
            commanders = self.ranking.keys()
        else:
            # keep commanders with diagnostic level > 1
            commanders = set()
            for commander, diag in self.diagnostic.items():
                if diag["warning_level"] > 0:
                    commanders.add(commander)
                    for s in diag["most_similars"]:
                        commanders.add(s[1])


        d = {}
        for commander in commanders:
            scores = self.ranking[commander]
            cd = {}
            for rank, record in scores.items():
                cd[rank] = ranking_record_to_dict(record)

            if commander in self.diagnostic:
                cd["diagnostic"] = self.diagnostic[commander]
            d[commander] = cd
        return d

    def get_commanders_dict(self):
        commanders = self.ranking.keys()
        d = {}
        for commander in commanders:
            score = first(self.ranking[commander])

            d[commander] = {"name": commander + " (" + score.alliance_short + ")"}
        return d

    def save_matching_images(self, commander1, commander2):

        def get_image(commander):
            record = first(self.ranking[commander])
            if record.original_name_image is not None:
                return record.original_name_image
            else:
                return record.name_image

        image1 = get_image(commander1)
        image2 = get_image(commander2)
        random_file_name = str(random.randint(0, 10000000))
        imageio.imwrite("name_matching/" + random_file_name + "_1.png", image1)
        imageio.imwrite("name_matching/" + random_file_name + "_2.png", image2)

    def merge(self, commanders_list):
        new_commander_name = commanders_list[0]
        for commander in commanders_list[1:]:
            self.ranking[new_commander_name].update(self.ranking[commander])
            del self.ranking[commander]
            del self.diagnostic[commander]
        self.diagnostic[new_commander_name] = {
            "warning_level": 0,
            "most_similars": [],
            "warnings": []
        }
        return new_commander_name

    def split(self, commander, rankings):
        commander_data = self.ranking[commander]
        new_commander_data = {}
        for rank in rankings:
            new_commander_data[rank] = commander_data[rank]
            del commander_data[rank]
        new_commander_name = new_commander_data[rankings[0]].name
        if new_commander_name == commander:
            new_commander_name = new_commander_name + str(random.randint(0, 1000))
        self.ranking[new_commander_name] = new_commander_data
        return new_commander_name

    def delete(self, commander):
        del self.ranking[commander]


class API:
    def __init__(self):
        self.ranking_data = None
        self.fusion_data = None
        self.window = None
        self.data_insertion_algorithm = None
        self.data_insertion_algorithm_options = None
        self.db_file = None
        self.ranking_files = None
        self.nation = None
        self.tmp_db_file = dbpath + "tmpDB.sqlite"
        self.proc_http_server = None

        import json
        with open("conf.json") as f:
            self.conf = json.load(f)

        self.start_http_server()

    def get_config(self):
        return self.conf

    def start_http_server(self):
        if self.proc_http_server is None:
            path = websitepath
            cmd = "python -m http.server --bind 127.0.0.1 --directory " + path
            import subprocess
            self.proc_http_server = subprocess.Popen(cmd.split())

    def init_ranking_data(self, ranking):
        self.ranking_data = RankingData(ranking)
        self.window.evaluate_js("move_to_ranking_fusion_page();")

    def ranking(self, method, *args):
        if self.ranking_data is None:
            raise Exception("ranking_data is not initialized")
        return getattr(self.ranking_data, method)(*args)

    def init_fusion_data(self, fusion_data):
        self.fusion_data = CommanderFusionData(fusion_data)
        self.window.evaluate_js("move_to_commander_fusion_page();")

    def fusion(self, method, *args):
        if self.fusion_data is None:
            raise Exception("fusion_data is not initialized")
        return getattr(self.fusion_data, method)(*args)

    def data_insertion_process(self, nation=None, dbfile=None, rankingfiles=None):
        if self.data_insertion_algorithm is None:
            self.nation = nation
            self.db_file = dbfile
            self.ranking_files = rankingfiles
            #remove tmp file if exists
            import os
            if os.path.exists(self.tmp_db_file):
                os.remove(self.tmp_db_file)
            # copy dbfile to tmp file
            import shutil
            shutil.copyfile(dbfile, self.tmp_db_file)

            self.data_insertion_algorithm_options = {}
            self.data_insertion_algorithm = insert_rankings_to_db_algorithm(nation, self.tmp_db_file, rankingfiles, self.data_insertion_algorithm_options, self)
        next(self.data_insertion_algorithm)

    def finalize_data_insertion(self):
        self.window.evaluate_js("move_to_finalize_data_insertion_page();")

    def clean_up_data_insertion(self, validate, backup_db_file):
        self.data_insertion_algorithm = None
        self.data_insertion_algorithm_options = None
        #if self.proc_http_server is not None:
        #    self.proc_http_server.kill()
        #    self.proc_http_server = None
        if validate:
            import shutil
            import os
            if backup_db_file:

                # date dd/mm/YY
                import datetime
                now = datetime.datetime.now()
                dt_string = now.strftime("%d_%m_%Y")
                # get filename from self.db_file
                filename = os.path.basename(self.db_file)
                shutil.copyfile(self.db_file, "old/" + filename + "_bak_" + dt_string)
            shutil.copyfile(self.tmp_db_file, self.db_file)
            for file in self.ranking_files:
                shutil.move(file, "old/" + os.path.basename(file))
            tmpFile1 = str(self.nation) + "_tmp_nation_data.pickle"
            tmpFile2 = str(self.nation) + "_tmp_nation_data2.pickle"
            if os.path.isfile(tmpFile2):
                os.remove(tmpFile2)
            if os.path.isfile(tmpFile1):
                os.rename(tmpFile1, tmpFile2)
        else:
            import os
            if os.path.isfile(self.tmp_db_file):
                os.remove(self.tmp_db_file)
            self.nation = None
            self.db_file = None
            self.ranking_files = None

    def test_data_insertion(self):
        self.start_http_server()

    def open_file_dialog(self):
        return self.window.create_file_dialog(webview.OPEN_DIALOG, allow_multiple=True)

    def find_latest_ranking_file(self, nation):
        import glob
        import os
        files = glob.glob(str(nation) + "_nation_ranking_[0-9]*.pickle")
        if len(files) == 0:
            return []
        def get_file_date(file: str):
            file = file[:-7]
            pos = file.find("_nation_ranking_")
            if pos == -1:
                return None
            date = file[pos + 16:]
            # convert to datetime
            import datetime
            return datetime.datetime.strptime(date, "%d-%m-%Y_%Hh-%Mm-%Ss")
        files = [(get_file_date(file), file) for file in files]
        files.sort(key=lambda x: x[0])
        latest = files[-1][1]
        latest_date = files[-1][0]
        res = [latest]

        files = glob.glob(str(nation) + "_cross_nation_ranking_*.pickle")
        if len(files) > 0:
            files = [(get_file_date(file), file) for file in files]
            files.sort(key=lambda x: x[0])

            latest_cross = files[-1][1]
            latest_cross_date = files[-1][0]
            # if same day return both
            if latest_date.date() == latest_cross_date.date():
                res.append(latest_cross)

        return res

    def notify(self, message):

        def callback(args):
            #list = []
            #win32gui.EnumWindows(lambda hwnd, result: result.append(hwnd), list)
#
            #list = [(hwnd,win32gui.GetWindowText(hwnd)) for hwnd in list]
            #list = [x for x in list if x[1] != ""]
            #print(list)
            #hwnd = win32gui.FindWindowEx(0, 0, 0, 'AOO data GUI')
            #print(hwnd)
            #win32gui.SetForegroundWindow(hwnd)
            window.on_top = True
            window.on_top = False
            pass

        toast("AOO data", message, audio='ms-winsoundevent:Notification.Default',
              on_click=callback)

    def compute_void_stats(self, nation):
        from void_stats import extract_void_stats
        from utils import get_date_of_first_strongest_commander_event_before
        date = get_date_of_first_strongest_commander_event_before(event_type="void")["date"]
        dbfile = dbpath + str(nation) + "DB.sqlite"
        result = extract_void_stats(dbfile, date)
        tsv = result.to_csv(index=False, sep="\t", encoding="utf-8", header=False)
        # send tsv to clipboard
        import pyperclip
        pyperclip.copy(tsv)
        return result.to_html(header="true", index=False, na_rep="")

api = API()
#api.init_ranking_data(data_ranking)
#api.init_fusion_data(data_fusion)

title = "AOO data GUI"

if __name__ == '__main__':
    window = webview.create_window(url="GUI/index.html", title=title, js_api=api, width=1200,
                                   height=1000, resizable=True, frameless=False, text_select=True)
    api.window = window
    webview.start(http_server=True, debug=True)