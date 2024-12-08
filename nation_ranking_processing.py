import cv2
import numpy as np

from Log import logger

from processRankingPage import RankingRecord

from utils import *
import pickle
import imageio

commander_rankings = [
    "commander_power",
    "commander_kill",
    "commander_city",
    "commander_officer",
    "commander_titan",
    "commander_warplane",
    "commander_island",
    "commander_merit",
    "commander_level",
    "commander_ke_frenzy",
    "commander_sc_frenzy",
    "commander_ke_void",
    "commander_sc_void",

    ]


alliance_rankings = ["alliance_power",
    "alliance_kill",
    "alliance_territory",
    "alliance_elite",
    "alliance_sc_frenzy",
    "alliance_sc_void",]


def merge_rankings_by_name(data, ranking_names, nation, is_alliance=False):
    if is_alliance:
        def get_name(record):
            return record.alliance_short.lower()
    else:
        def get_name(record):
            return record.name

    if not is_alliance and "alliance_members" in data:
        d = data["alliance_members"]
        data_by_name = {}
        start = 0
        for alliance in d:
            for member in d[alliance]:
                member_data = d[alliance][member]
                name = member_data.name

                #im2 = member_data.name_image
                #im2 = cv2.resize(im2, (int(im2.shape[1] * 1.07), int(im2.shape[0] * 1.07)))
                #im2 = (((im2 / 255) ** 1.3) * 255).astype(np.uint8)
                #member_data.name_image = im2

                rnks = {}
                data_by_name[name] = rnks
                rnks["commander_reputation"] = RankingRecord(-1, name, alliance, member_data.reputation, member_data.name_image, nation, 2)
                rnks["commander_power"] = RankingRecord(-1, name, alliance, member_data.battle_power, member_data.name_image, nation, 2)
                rnks["commander_kill"] = RankingRecord(-1, name, alliance, member_data.kills, member_data.name_image, nation, 2)
                rnks["commander_loss"] = RankingRecord(-1, name, alliance, member_data.losses, member_data.name_image, nation, 2)
                rnks["commander_island"] = RankingRecord(-1, name, alliance, member_data.luxuriousness, member_data.name_image, nation, 2)
                rnks["commander_merit"] = RankingRecord(-1, name, alliance, member_data.merit_value, member_data.name_image, nation, 2)
                rnks["commander_city"] = RankingRecord(-1, name, alliance, member_data.city_level, member_data.name_image, nation, 2)
                #class RankingRecord(RecordClass):
                #    rank: int
                #    name: str
                #    alliance_short: str
                #    score: int
                #    name_image: np.ndarray
                #    nation: int

                #class CityStats(RecordClass):
                #    name: str
                #    battle_power: int
                #    reputation: int
                #    kills: int
                #    losses: int
                #    luxuriousness: int
                #    merit_value: int
                #    name_image: np.ndarray

    else:
        start = 1
        data1 = data[ranking_names[0]]
        data_by_name = {get_name(data1[x]): {ranking_names[0]: data1[x]} for x in data1 if (data1[x].nation == 0 or data1[x].nation == nation)}

    stats = {}

    for ranking in ranking_names[start:]:
        if ranking not in data:
            continue
        logger.logSection("Early merge: " + ranking)
        data2 = data[ranking]
        data2_names = {get_name(data2[x]): {ranking: data2[x]} for x in data2 if (data2[x].nation == 0 or data2[x].nation == nation)}

        for name2 in data2_names:
            data2_names[name2][ranking].original_name_image = data2_names[name2][ranking].name_image
            image2 = proc_image_with_alliance(data2_names[name2][ranking].name_image)
            data2_names[name2][ranking].name_image = image2
            data2_names[name2][ranking].image_version = 2

        inserted = []
        for name in data2_names:
            if name in data_by_name:
                if ranking not in data_by_name[name]:
                    data_by_name[name].update(data2_names[name])
                else:
                    data_by_name[name][ranking].rank = data2_names[name][ranking].rank

                    #if data2_names[name][ranking].score > 0 and data_by_name[name][ranking].score <= 0:
                    if data2_names[name][ranking].score > data_by_name[name][ranking].score:
                        data_by_name[name][ranking].score = data2_names[name][ranking].score
                inserted.append(name)

        for name in inserted:
            del data2_names[name]



        inserted = []
        for name2 in data2_names:
            image2 = data2_names[name2][ranking].name_image

            if name2 not in stats:
                stats[name2] = SortedFixSizedList(5)

            matches = []
            for name1 in data_by_name:
                if ranking not in data_by_name[name1] or data_by_name[name1][ranking].rank == -1:

                    image1 = data_by_name[name1][next(iter(data_by_name[name1]))].name_image
                    #if name1 == "mw? Xhoan2k=—=——":
                    #    print("op")
                    #    imshow(image2)
                    #    imshow(image1)
                    #    imshow(image2)
                    #    imshow(image1)
                    res = match_image_name(image1, image2, return_score=True)#, size_diff_thr=100)
                    stats[name2].add(res, name1)
                    if res > 0.82: #0.85
                        matches.append((name1, res))

            if len(matches) > 0:
                matches.sort(key=lambda x: x[1], reverse=True)
                name1 = matches[0][0]
                image1 = data_by_name[name1][next(iter(data_by_name[name1]))].name_image
                # print("match:", name1, " ---> ", name2)
                logger.logEntryMatch(name1, image1, name2, image2)
                if ranking not in data_by_name[name1]:
                    data_by_name[name1].update(data2_names[name2])
                else:
                    data_by_name[name1][ranking].rank = data2_names[name2][ranking].rank
                    if data2_names[name2][ranking].score > 0 and data_by_name[name1][ranking].score <= 0:
                        data_by_name[name1][ranking].score = data2_names[name2][ranking].score
                inserted.append(name2)


        for name in inserted:
            del data2_names[name]

        logger.write("<h3>Non matched commanders:</h3>")
        for name2 in data2_names:
            logger.logEntry(name2, data2_names[name2][ranking].name_image)
            data_by_name[name2] = data2_names[name2]

        #print(data2_names)

    if not is_alliance:
        manual_matches = [ # ('<6 7OO+ ge,', '꧁༒クロロ༒꧂'),
            #('eS EY AF NOG,', '꧁ヒソカ༒NO.4꧂'),
            #('~}-Shadolt}-', '༒Shadoll༒'),
            #('gg-—~ 7 A.Cobos—=——','▄︻デA.Cobos══━一'),
            #('gg-—~ 7 Rok2000————','▄︻デRok2000══━一'),
            #('gp ROK2000————', '▄︻デRok2000══━一'),
            #('gg-—~ 7 Xhoan2==——','▄︻デXhoan2══━一'),
            #('gy—~ 7 Xhoan2=————', '▄︻デXhoan2══━一'),
            #('gy—~ 7 C9H13N—=——', '▄︻デC9H13N══━一'),
            #('<6 7OO+ ge,','꧁༒クロロ༒꧂'),
            #('eG kratos FS,','eG kratos FS,')
        ]

        for old_name, new_name in manual_matches:
            data_old = data_by_name[old_name]
            #remove old name
            del data_by_name[old_name]
            # update dict in new name with old data dict
            data_by_name[new_name].update(data_old)

    diags = {}
    if not is_alliance:
        detailed_alliances = list(data["alliance_members"].keys()) if "alliance_members" in data else []
        detailed_alliances = [alliance.lower() for alliance in detailed_alliances]
        # diagnostics
        #Sty F< Ag,
        for name in data_by_name:
            d = {"warning_level": 0, "warnings": [], "most_similars": stats[name].data if name in stats else []}
            alliance_short = first(data_by_name[name]).alliance_short.lower()
            if (alliance_short != "") and (alliance_short in detailed_alliances) and ("commander_loss" not in data_by_name[name]):
                d["warning_level"] = max(d["warning_level"], 10)
                d["warnings"].append("Commander is in detailed alliance but core stats are missing.")

            if name in stats and stats[name].data[-1][0] < 0.88:
                d["warning_level"] = max(d["warning_level"], 1)
                d["warnings"].append("Matching level is low.")

            if name in stats and 0.82 > stats[name].data[-1][0] > 0.5:
                d["warning_level"] = max(d["warning_level"], 5)
                d["warnings"].append("Close matched rejected.")

            diags[name] = d


        #dump commander_ranking to file
        #with open("commander_ranking.pickle", "wb") as f:
        #    pickle.dump({"commander_ranking": data_by_name, "diagnostic": diags}, f)
        #exit()

    return data_by_name, diags

def extract_commander_ranking_info(data, nation):
    return merge_rankings_by_name(data, commander_rankings, nation)

def extract_alliance_ranking_info(data, nation):
    return merge_rankings_by_name(data, alliance_rankings, nation, is_alliance=True)








def test_two_lines():
    files = [("matching_no4",False),("kumakoro", True), ("ringo", False), ("apple", False), ("matching_wild_hunt", False)]

    for f, v in files:
        im = imageio.imread(f + "2.png")
        print(f, has_two_lines_alliance_removal(im), v)
        #imshow(im, cmap="gray")

if __name__ == "__main__":
    files = ["matching_cn6npruka","matching_soibac","matching_botafogo","matching_kazu","matching_un199","matching_esal","matching_yana","matching_mickoramus","matching_GP","matching_lkr001","matching_kemist","matching_dracaris","matching_bayok","matching_kohedon","matching_ricardo", "matching_no4","matching_GIS","matching_zzzxxx","matching_xhoan2","matching_wild_hunt","kumakoro" ]#,"ringo","crea","hwi","devil","wcrea","mick","apple","ringoo","parz"]

    import glob
    files = glob.glob("name_matching/auto/*1.png")
    files = [f[:-5] for f in files]
    import imageio.v3 as imageio

    #test_two_lines()
    #exit()

    if False:
        f = "matching_soibac"
        im1 = imageio.imread(f + "1.png")
        im2 = imageio.imread(f + "2.png")

        im2 = proc_image_with_alliance(im2, rescale=False)
        imshow(im2, cmap="gray")
        print(im1.shape, im2.shape)

        im2 = cv2.resize(im2, (int(im2.shape[1] * 1.2), int(im2.shape[0] * 1.01)))
        imshow(im1, cmap="gray")
        imshow(im2, cmap="gray")
        print(f, "->", match_image_name(im1, im2, return_score=True))
        exit()


    if True:
        images1 = [imageio.imread(f + "1.png") for f in files]
        images2 = [imageio.imread(f + "2.png") for f in files]
        all = []
        res_per_image = [[] for _ in range(len(files))]
        recompute = True
        if recompute:
            for x in np.linspace(1.0, 1.4, 30):
                for y in np.linspace(1.00, 1.4, 30):
                    res= []
                    names = []
                    for i in range(len(files)):

                        im1 = images1[i].copy()
                        #im1 = (((im1/255)**0.5)*255).astype(np.uint8)
                        #imshow(im1, cmap="gray")
                        im2 = images2[i].copy()
                        im2 = proc_image_with_alliance(im2, rescale=False)
                        im2 = cv2.resize(im2, (int(im2.shape[1] * x), int(im2.shape[0]*y)), interpolation=cv2.INTER_AREA)
                        #imshow(im2, cmap="gray")
                        #im2 = (((im2 / 255)**1.5)*255).astype(np.uint8)
                        #imshow(im1, cmap="gray")
                        #imshow(im2, cmap="gray")
                        #print(im1.shape, im2.shape, im2.max())
                        #print(f, "->", match_image_name(im1, im2, return_score=True), end=", ")
                        score = match_image_name(im1, im2, return_score=True)
                        res.append(score)
                        res_per_image[i].append((score, x, y))

                    #exit()
                    amin = np.argmin(res)
                    all.append((x, y, res[amin], files[amin]))

            print("min-max all")
            all = sorted(all, key=lambda x: -x[2])
            print(all[:3])

            print("best per image")
            x = []
            y = []
            for i in range(len(files)):
                best = sorted(res_per_image[i], key=lambda x: -x[0])[0]
                print(files[i], best)
                im2 = images2[i].copy()
                im2 = proc_image_with_alliance(im2, rescale=False)
                x.append((im2.shape[0], im2.shape[1], im2.shape[0] / im2.shape[1]))
                y.append((best[1], best[2]))

            x = np.array(x)
            y = np.array(y)

            # save x, y
            pickle.dump((x, y), open("resizing_xy.pkl", "wb"))
        else:
            x, y = pickle.load(open("resizing_xy.pkl", "rb"))

        import sklearn as sk
        from sklearn import tree
        from sklearn.ensemble import RandomForestRegressor
        # fit and test decision tree regression model on x, y
        model = tree.DecisionTreeRegressor(random_state=0, max_depth=5)
        #model = RandomForestRegressor(random_state=0, max_depth=6, n_estimators=100)
        model.fit(x, y)

        # plot results versus ground truth
        import matplotlib.pyplot as plt
        plt.scatter(x[:, 0], y[:, 0], label="ground truth")
        plt.scatter(x[:, 0], model.predict(x)[:, 0], label="prediction")
        plt.legend()
        plt.show()


        # predict y given x
        y_pred = model.predict(x)
        print(y_pred - y)


        #save model
        resize_model_regressor = model
        pickle.dump(model, open("resizing_decision_tree.pkl", "wb"))





    if True:



        res= []
        names = []
        for f in files:

            im1 = imageio.imread(f + "1.png")
            im2 = imageio.imread(f + "2.png")
            im2 = proc_image_with_alliance(im2)
            #boost contrast
            #im2 = (((im2 / 255)**1.2)*255).astype(np.uint8)
            # small dilationi
            #im2 = cv2.erode(im2, np.ones((3, 3), np.uint8), iterations=1)
            #im2 = (((im2 / 255)**1.5)*255).astype(np.uint8)
            #imshow(im1, cmap="gray")
            #imshow(im2, cmap="gray")
            #print(im1.shape, im2.shape, im2.max())
            r = match_image_name(im1, im2, return_score=True)
            print(f, "->", r, end=", ")
            res.append(r)

            #exit()
        amin = np.argmin(res)
        print("global")
        print((res[amin], files[amin]))

