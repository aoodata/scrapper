from datetime import datetime, timedelta

#import imageio
#import numpy as np
import sqlite3
#import pickle
import imageio.v3 as iio
import random

random.seed(1)
from utils import *

from Log import logger
#from processRankingPage import preproc_text_image
from nation_ranking_processing import extract_alliance_ranking_info, extract_commander_ranking_info, match_image_name


def encode_image(img):
    return iio.imwrite("<bytes>", img, extension=".png")


def decode_image(data):
    return iio.imread(data)


def first(d):
    return d[next(iter(d))]


def insert_alliances_to_db(db_connection, alliance_ranking, updateNames=False):
    alliances = {}
    cursor = db_connection.cursor()
    logger.logSection("Alliances DB merge")

    alliances_db = cursor.execute("SELECT id, name_short, name_long, name_image FROM alliances").fetchall()
    alliances_db = {x[1]: (x[0], x[1], x[2], decode_image(x[3]) if x[0] != 0 else None) for x in alliances_db}
    used_alliances_db = set()
    used_alliances_db.add("UNKNOWN")
    alliance_to_insert = set(alliance_ranking.keys())

    for alliance in alliance_ranking:
        exist = cursor.execute(
            "SELECT id, name_short, name_long, name_image FROM alliances WHERE lower(name_short) = ?",
            (alliance,)).fetchone()
        if exist:
            alliances[alliance] = (exist[0], alliance, exist[2], decode_image(exist[3]))
            alliance_to_insert.remove(alliance)
            used_alliances_db.add(alliance)

    found = []
    for alliance1 in alliance_to_insert:
        for alliance2 in alliances_db:
            if alliance2 not in used_alliances_db:
                image1 = first(alliance_ranking[alliance1]).name_image[:, 20:]
                image2 = alliances_db[alliance2][3][:, 20:]
                if match_image_name(image1, image2):
                    logger.logEntryMatch(alliance1, image1, alliance2, image2)
                    found.append(alliance1)
                    alliances[alliance1] = alliances_db[alliance2]
                    used_alliances_db.add(alliance2)
                    break
    for alliance in found:
        alliance_to_insert.remove(alliance)

    if updateNames:
        # update name_long and name_image
        for alliance in alliances:
            record = first(alliance_ranking[alliance])
            cursor.execute("UPDATE alliances SET name_long = ?, name_image = ? WHERE id = ?",
                           (record.name, encode_image(record.name_image), alliances[alliance][0]))

    logger.logSection("Alliances DB merge => insert new")
    for alliance in alliance_to_insert:
        logger.write(alliance + "<br>")
        logger.logEntry(alliance, first(alliance_ranking[alliance]).name_image, alliance_ranking[alliance])
        record = first(alliance_ranking[alliance])
        cursor.execute("INSERT INTO alliances (name_short, name_long, name_image) VALUES (?,?,?)",
                       (alliance, record.name, encode_image(record.name_image, )))
        alliances[alliance] = (cursor.lastrowid, alliance, record.name, record.name_image)

    return alliances


def insert_alliances_rankings_to_db(db_connection, alliance_ranking, alliances, timestamp):
    rankings = {}
    cursor = db_connection.cursor()
    for alliance in alliance_ranking:
        for ranking in alliance_ranking[alliance]:
            rankings[ranking] = None

    for ranking in rankings:
        ranking_id = cursor.execute("SELECT id FROM data_collection_types WHERE name = ?", (ranking,)).fetchone()
        ranking_id = ranking_id[0]
        cursor.execute("INSERT INTO data_collections (type_id, date) VALUES (?,?)", (ranking_id, timestamp))
        rankings[ranking] = cursor.lastrowid

    for alliance in alliance_ranking:
        for ranking in alliance_ranking[alliance]:
            record = alliance_ranking[alliance][ranking]
            cursor.execute(
                "INSERT INTO alliance_ranking_data (data_collection_id, alliance_id, rank, value) VALUES (?,?,?,?)",
                (rankings[ranking], alliances[alliance][0], record.rank, record.score))

    return rankings


def insert_commanders_rankings_to_db(db_connection, commander_ranking, commanders, timestamp):
    cursor = db_connection.cursor()
    rankings = {}

    for commander in commander_ranking:
        for ranking in commander_ranking[commander]:
            rankings[ranking] = None

    for ranking in rankings:
        ranking_id = cursor.execute("SELECT id FROM data_collection_types WHERE name = ?", (ranking,)).fetchone()[0]
        cursor.execute("INSERT INTO data_collections (type_id, date) VALUES (?,?)", (ranking_id, timestamp))
        rankings[ranking] = cursor.lastrowid

    for commander in commander_ranking:
        for ranking in commander_ranking[commander]:
            record = commander_ranking[commander][ranking]
            cursor.execute(
                "INSERT INTO commander_ranking_data (data_collection_id, commander_id, rank, value) VALUES (?,?,?,?)",
                (rankings[ranking], commanders[commander][0], record.rank, record.score))

    return rankings


from typing import NamedTuple
from recordclass import RecordClass


class RankingInfo(RecordClass):
    name: int
    id: int
    weight: float
    increasing: bool
    latest_collection_id: int
    min_significant_change: float


class CommanderScore(NamedTuple):
    score: float
    rank: int


class CommanderScoreHelper:

    def __init__(self, cursor):
        self.cursor = cursor
        self.rankings_info = [
            RankingInfo("commander_power", 1, 0, False, -1, 0),
            RankingInfo("commander_kill", 2, 1, True, -1, 500000),
            RankingInfo("commander_city", 3, 0, True, -1, 0),
            RankingInfo("commander_officer", 4, 0.05, False, -1, 20000),
            RankingInfo("commander_titan", 5, 0.5, False, -1, 5000),
            RankingInfo("commander_island", 6, 0.5, True, -1, 100),
            RankingInfo("commander_merit", 7, 0.8, True, -1, 200),
            RankingInfo("commander_level", 8, 0, True, -1, 58),
            # RankingInfo("commander_reputation", 24, 0.5, True, -1, 500),
            RankingInfo("commander_loss", 25, 0.05, True, -1, 200000),
            RankingInfo("commander_warplane", 23, 0.5, True, -1, 2000),
        ]

        self.used_rankings = {x.name: x for x in self.rankings_info[1:]}

        def find_latest_ranking_ids():
            for ranking in self.used_rankings:
                ranking_id = self.cursor.execute(
                    "SELECT id FROM data_collections WHERE data_collections.type_id = ? order by date desc limit 1",
                    (self.used_rankings[ranking].id,)).fetchone()
                if ranking_id is not None:
                    self.used_rankings[ranking].latest_collection_id = ranking_id[0]
                else:
                    self.used_rankings[ranking].latest_collection_id = -1

        find_latest_ranking_ids()

        def find_date_of_latest_ranking():
            ranking = self.used_rankings["commander_kill"]

            date = self.cursor.execute(
                "SELECT date FROM data_collections WHERE data_collections.type_id = ? order by date desc limit 1",
                (ranking.id,)).fetchone()
            if date is not None:
                return date[0]
            return 0

        self.date_of_latest_ranking = find_date_of_latest_ranking()

        self.latest_ref_date = self.date_of_latest_ranking - 30 * 24 * 3600 # 30 days before the latest ranking

        self.scores_db = {}

    def get_latest_scores(self, commander):
        if commander in self.scores_db:
            return self.scores_db[commander]
        else:
            sc = {}
            for ranking in self.used_rankings:
                #r = self.cursor.execute(
                #    "SELECT value, rank FROM commander_ranking_data WHERE  commander_ranking_data.commander_id = ? AND commander_ranking_data.data_collection_id = ?",
                #    (commander, self.used_rankings[ranking].latest_collection_id)).fetchone()
                r = self.cursor.execute(
                    "SELECT value, rank FROM commander_ranking_data, data_collections "
                    "WHERE  commander_ranking_data.commander_id = ? "
                    "AND data_collections.type_id = ? "
                    "AND data_collections.id = commander_ranking_data.data_collection_id "
                    "AND data_collections.date >= ?"
                    "ORDER BY data_collections.date DESC LIMIT 1",
                    (commander, self.used_rankings[ranking].id, self.latest_ref_date)).fetchone()
                if r is not None:
                    sc[ranking] = CommanderScore(r[0], r[1])
            self.scores_db[commander] = sc
            return sc

    def missed_ranking_penalty(self, rank):
        import math
        if rank == -1:
            return 0.2
        return 0.5 / (1 + math.exp((rank - 70) / 3))

    def commander_relative_difference(self, commander_to_insert, commander_db,
                                      missed_penalty=0.03):  # missed_penalty=0.03
        relative_difference = 0
        total_weight = 0
        missed = 0
        num_matched = 0
        for ranking in self.used_rankings:
            if ranking in commander_to_insert and ranking in commander_db:
                num_matched += 1
                div = max(commander_to_insert[ranking].score, commander_db[ranking].score)
                if div == 0:
                    print("Commander scores are both 0!", ranking, commander_to_insert[ranking])
                    div = 1
                diff = abs(commander_to_insert[ranking].score - commander_db[ranking].score)
                total_weight += self.used_rankings[ranking].weight
                if diff < self.used_rankings[ranking].min_significant_change:
                    continue
                relative_difference += self.used_rankings[ranking].weight * diff / div

                if self.used_rankings[ranking].increasing:
                    if commander_to_insert[ranking].score < commander_db[ranking].score:
                        return float("inf")
            elif ranking in commander_to_insert and ranking not in commander_db and ranking != "commander_city":
                missed += self.missed_ranking_penalty(commander_to_insert[ranking].rank)
            elif ranking in commander_db and ranking not in commander_to_insert:
                missed += self.missed_ranking_penalty(commander_db[ranking].rank)
        if total_weight == 0:
            return float("nan")
        return relative_difference + missed + missed_penalty * (len(self.used_rankings) - num_matched)


from utils import SortedFixSizedList


def stat_based_commander_matching(cursor, commander_ranking, non_matched_commanders_db, commanders_to_insert,
                                  scoreHelper):
    class Edge(NamedTuple):
        commander_to_insert: dict
        commander_db: dict
        difference: float

    edges = []
    stats = {}

    for commander1 in commanders_to_insert:
        stats[commander1] = SortedFixSizedList(3)
        for commander2 in non_matched_commanders_db:
            diff = scoreHelper.commander_relative_difference(commander_ranking[commander1],
                                                             scoreHelper.get_latest_scores(
                                                                 non_matched_commanders_db[commander2][0]))
            if diff < 0.4:  # 0.8
                edges.append(Edge(commander1, commander2, diff))
            if diff < 100:
                stats[commander1].add(-diff, commander2)

    edges.sort(key=lambda x: x.difference)


    matched_commanders_db = set()
    matched_commanders_to_insert = {}
    for edge in edges:
        if edge.commander_to_insert not in matched_commanders_to_insert and edge.commander_db not in matched_commanders_db:
            commander1_alliance = first(commander_ranking[edge.commander_to_insert]).alliance_short
            commander2_alliance = non_matched_commanders_db[edge.commander_db][2]
            logger.logEntryStatsMatch(edge.commander_to_insert,
                                      commander_ranking[edge.commander_to_insert],
                                      edge.commander_db,
                                      scoreHelper.get_latest_scores(non_matched_commanders_db[edge.commander_db][0], ),
                                      commander1_alliance,
                                      commander2_alliance, )
            matched_commanders_to_insert[edge.commander_to_insert] = (edge.commander_db, -edge.difference)
            matched_commanders_db.add(edge.commander_db)
            commanders_to_insert.remove(edge.commander_to_insert)
            # del non_matched_commanders_db[edge.commander_db]

    return matched_commanders_to_insert, commanders_to_insert, stats



def match_commanders_to_db(db_connection, commander_ranking, alliances, timestamp, detailed_alliances):
    import math
    cursor = db_connection.cursor()
    scoreHelper = CommanderScoreHelper(cursor)
    logger.logSection("Commanders DB merge")
    #from nation_ranking_processing import proc_image_with_alliance

    commanders = {}
    commanders_to_insert = set(commander_ranking.keys())
    commanders_db = cursor.execute(
        "SELECT commanders.id, commanders.canonical_name, alliances.name_short,  commanders.name_image, commanders.version  FROM commanders, alliances WHERE commanders.alliance_id = alliances.id").fetchall()
    commanders_db = {x[1]: (x[0], x[1], x[2], decode_image(x[3]), x[4]) for x in commanders_db}

    commanders_db_scores = {}
    for commander in commanders_db:
        commanders_db_scores[commander] = scoreHelper.get_latest_scores(commanders_db[commander][0])

    used_commanders_db = set()

    score_diff_rejection_threshold = 0.5  # 1.25

    for commander in commander_ranking:

        if commander.strip() in commanders_db:
            commanderScore = commander_ranking[commander]
            commanderDBScore = scoreHelper.get_latest_scores(commanders_db[commander.strip()][0])
            diff = scoreHelper.commander_relative_difference(commanderScore, commanderDBScore)
            # test if diff is nan

            if math.isnan(diff) or diff >= score_diff_rejection_threshold: # test if diff is nan
                logger.write("Name match: commander " + commander + " is too different from the one in the DB: " + str(
                    diff) + "<br>")
                commander1_alliance = first(commanderScore).alliance_short
                commander2_alliance = commanders_db[commander.strip()][2]
                # scoreHelper.commander_relative_difference(commanderScore, commanderDBScore)
                logger.logEntryStatsMatch(commander,
                                          commander_ranking[commander],
                                          commander,
                                          scoreHelper.get_latest_scores(commanders_db[commander.strip()][0]),
                                          commander1_alliance,
                                          commander2_alliance, )
                continue
            commanders[commander] = commanders_db[commander.strip()]
            commanders_to_insert.remove(commander)
            used_commanders_db.add(commander)

    found = []
    for commander1 in commanders_to_insert:
        matches = []
        commander1_ranking = first(commander_ranking[commander1])
        commander1_image = commander1_ranking.name_image
        commander1_image_version = commander1_ranking.image_version
        commander1_alliance = commander1_ranking.alliance_short
        #if commander1_image_version == 1:
        #    commander_1_image_preproc1 = remove_alliance_name(commander1_image)
        #    commander_1_image_preproc2 = proc_image_with_alliance(commander1_image)
        for commander2 in commanders_db:
            if commander2 not in used_commanders_db:
                commander2_image = commanders_db[commander2][3]
                commander2_image_version = commanders_db[commander2][4]
                # image version 1 was removed
                if commander1_image_version == 2 and commander2_image_version == 2:
                    image1 = commander1_image
                    image2 = commander2_image
                #elif commander1_image_version == 1 and commander2_image_version == 2:
                #    image1 = commander_1_image_preproc2
                #    image2 = commander2_image
                #elif commander1_image_version == 2 and commander2_image_version == 1:
                #    image1 = commander1_image
                #    image2 = proc_image_with_alliance(commander2_image)
                #elif commander1_image_version == 1 and commander2_image_version == 1:
                #    image1 = commander_1_image_preproc1
                #    image2 = remove_alliance_name(commander2_image)
                else:
                    raise Exception("Image version not supported")

                try:
                    v = match_image_name(image1, image2, return_score=True)
                    matches.append((commander2, v))
                except Exception as e:

                    print("Error matching images", commander1, commander2, commander1_image_version,
                          commander2_image_version)
                    imshow(image1)
                    imshow(image2)
                    # imshow(first(commander_ranking[commander1]).name_image)
                    # imshow(commanders_db[commander2][3])
                    raise Exception("Error matching images", e)
        if len(matches) > 0:
            matches.sort(key=lambda x: x[1], reverse=True)
            if matches[0][1] > 0.9:
                commander2 = matches[0][0]
                commander2_alliance = commanders_db[commander2][2]
                image2 = commanders_db[commander2][3] #remove_alliance_name(commanders_db[commander2][3])

                score = scoreHelper.commander_relative_difference(commander_ranking[commander1],
                                                                  scoreHelper.get_latest_scores(
                                                                      commanders_db[commander2][0]))
                if math.isnan(score) or score >= score_diff_rejection_threshold:  # 0.55:
                    logger.logEntryMatch(commander1, image1, commander2, image2)
                    logger.write(
                        "======>> Image match: commander " + commander1 + " is too different from the one in the DB: " + str(
                            score) + "<br>")
                    logger.logEntryStatsMatch(commander1,
                                              commander_ranking[commander1],
                                              commander2,
                                              scoreHelper.get_latest_scores(commanders_db[commander2][0]),
                                              commander1_alliance,
                                              commander2_alliance, )
                else:
                    logger.logEntryMatch(commander1, image1, commander2, image2, commander1_alliance,
                                         commander2_alliance)
                    found.append(commander1)
                    commanders[commander1] = commanders_db[commander2]
                    used_commanders_db.add(commander2)

    for commander in found:
        commanders_to_insert.remove(commander)

    # try to match remaining commanders with stats
    stats_match = True
    matched_commanders_to_insert = {}
    stats = {}
    commander_to_insert_scores = {}
    commander_db_scores = {}
    non_matched_commanders_db = {x: commanders_db[x] for x in commanders_db if x not in used_commanders_db}

    if stats_match:
        logger.logSection("Commander stats matching")
        if len(commanders_to_insert) > 0 and len(non_matched_commanders_db) > 0:
            matched_commanders_to_insert, commanders_to_insert, stats = (
                stat_based_commander_matching(cursor, commander_ranking, non_matched_commanders_db,
                                              commanders_to_insert, scoreHelper))
            # macthed_commanders_to_insert is a dict of commander_name_in_ranking_data -> (commander_ranking_data, commander_name_in_db, score_diff)

        def get_score_commander_to_insert(commander):
            score = commander_ranking[commander]
            res = {}
            for ranking in score:
                res[ranking] = score[ranking].score
            return {"scores":res, "alliance": first(score).alliance_short}

        for commander in commanders_to_insert:
            if commander not in commander_to_insert_scores:
                commander_to_insert_scores[commander] = get_score_commander_to_insert(commander)

        for commander in matched_commanders_to_insert:
            if commander not in commander_to_insert_scores:
                commander_to_insert_scores[commander] = get_score_commander_to_insert(commander)

        for commander in stats:
            data = stats[commander].data
            stats[commander] = data
            for _, commander_db in data:
                if commander_db not in commander_db_scores:
                    d = {}
                    score = scoreHelper.get_latest_scores(commanders_db[commander_db][0])
                    for ranking in score:
                        d[ranking] = score[ranking].score
                    commander_db_scores[commander_db] = {"scores":d,"alliance": commanders_db[commander_db][2]}

        # with open("commander_fusion.pickle", "wb") as f:
        #    pickle.dump({
        #        "matched_commanders_to_insert": matched_commanders_to_insert,
        #        "commanders_to_insert": commanders_to_insert,
        #        "matching_stats": stats,
        #        "commander_to_insert_scores": commander_to_insert_scores,
        #        "commander_db_scores": commander_db_scores}, f)
        #    exit()
    return commanders, matched_commanders_to_insert, commanders_to_insert, stats, commander_to_insert_scores, commander_db_scores, non_matched_commanders_db


def insert_new_commanders_and_update_matched(db_connection, commanders, commander_ranking, alliances,
                                             detailed_alliances,
                                             timestamp, matched_commanders_to_insert, commanders_to_insert,
                                             non_matched_commanders_db):
    cursor = db_connection.cursor()
    if len(matched_commanders_to_insert) > 0:
        for commander in matched_commanders_to_insert:
            # commander_record_in_db, commander_name_in_db, diff = matched_commanders_to_insert[commander]
            commander_name_in_db, diff = matched_commanders_to_insert[commander]
            commander_record_in_db = non_matched_commanders_db[commander_name_in_db]
            one_record = first(commander_ranking[commander])
            commander_id = commander_record_in_db[0]
            commander_canonical_name = commander_record_in_db[1]
            # store old commander name
            cursor.execute("INSERT INTO commander_names (commander_id, name, date) VALUES (?, ?, ?)",
                           (commander_id, commander_canonical_name, timestamp))
            # update commander name
            cursor.execute("UPDATE commanders SET canonical_name = ?, name_image = ?, version = ? WHERE id = ?", (
                one_record.name, encode_image(one_record.name_image), one_record.image_version, commander_id))
            commanders[commander] = commander_record_in_db

    # anual_match = {
    #   #    "Ohhtani": commanders_db["Ohhhtani"],
    #

    # or commander in manual_match:
    #   commanders[commander] = manual_match[commander]
    #   commanders_to_insert.remove(commander)

    #   one_record = first(commander_ranking[commander])
    #   alliance_short = one_record.alliance_short.lower()
    #   if alliance_short in alliances:
    #       alliance_id = alliances[alliance_short][0]
    #   else:
    #       alliance_id = 0
    #   # update name image
    #   cursor.execute(
    #       "UPDATE commanders SET canonical_name = ?, alliance_id = ?, name_image = ?, version = ? WHERE id = ?", (
    #           commander, alliance_id, encode_image(one_record.name_image), one_record.image_version,
    #           commanders[commander][0]))

    # update alliances if needed
    for commander in commanders:
        one_record = first(commander_ranking[commander])
        alliance_short = one_record.alliance_short.lower()
        if alliance_short in alliances:
            alliance_id = alliances[alliance_short][0]
        else:
            alliance_id = 0
        # update name image
        # cursor.execute("UPDATE commanders SET canonical_name = ?, alliance_id = ?, name_image = ?, version = ? WHERE id = ?", (commander, alliance_id, encode_image(one_record.name_image), one_record.image_version, commanders[commander][0]))

        if commanders[commander][2].lower() != alliance_short:
            cursor.execute("UPDATE commanders SET alliance_id = ? WHERE id = ?",
                           (alliance_id, commanders[commander][0]))

    logger.logSection("Commanders DB merge - insert new commanders")
    for commander in commanders_to_insert:
        record = first(commander_ranking[commander])
        # imshow(record.name_image)
        alliance_short = record.alliance_short.lower()
        if alliance_short in detailed_alliances and "commander_loss" not in commander_ranking[commander]:
            logger.write(
                "<b style='color:red;'>Warning:</b> Commander " + commander + " is in " + alliance_short + "but detailed member stats are missing! => failed merge!<br>")

        logger.logEntry(commander, record.name_image, commander_ranking[commander], alliance_short)

        if alliance_short in alliances:
            alliance_id = alliances[alliance_short][0]
        else:
            alliance_id = 0
        cursor.execute("INSERT INTO commanders (canonical_name, alliance_id, name_image, version) VALUES (?,?,?,?)",
                       (record.name.strip(), alliance_id, encode_image(record.name_image), record.image_version))
        commanders[commander] = (
            cursor.lastrowid, commander, record.alliance_short, record.name_image, record.image_version)

    return commanders


def insert_sc_to_db(db_connection, data, event_type, alliance_data_collection_ids, commander_data_collection_ids,
                    timestamp):
    cursor = db_connection.cursor()
    opponent = data["other_nation_name"]
    nation_score = data["nation_sc_" + event_type + "_score"]
    opponent_score = data["other_nation_sc_" + event_type + "_score"]
    alliance_sc_ranking_id = alliance_data_collection_ids["alliance_sc_" + event_type]
    commander_sc_ranking_id = commander_data_collection_ids["commander_sc_" + event_type]
    commander_ke_ranking_id = commander_data_collection_ids["commander_ke_" + event_type]

    # convert timestamp to datetime
    date = datetime.fromtimestamp(timestamp)
    # find first sunday before date
    date = date - timedelta(days=date.weekday() + 1)
    # only keep year, month and day
    date = date.replace(hour=0, minute=0, second=0, microsecond=0)
    # convert back to timestamp
    timestamp = int(date.timestamp())

    cursor.execute(
        "INSERT INTO " + event_type + " (opponent, nation_score, opponent_score, date, alliance_sc_ranking_id, commander_sc_ranking_id, commander_ke_ranking_id, date) VALUES (?,?,?,?,?,?,?,?)",
        (opponent, nation_score, opponent_score, timestamp, alliance_sc_ranking_id, commander_sc_ranking_id,
         commander_ke_ranking_id, timestamp))


def insert_rankings_to_db_algorithm(nation, db_file, ranking_files, options, gui):
    data = {}
    for filename in ranking_files:
        with open(filename, "rb") as f:
            data.update(pickle.load(f))


    nation = int(nation)
    timestamp = data["date"]

    detailed_alliances = list(data["alliance_members"].keys()) if "alliance_members" in data else []
    detailed_alliances = [alliance.lower() for alliance in detailed_alliances]



    ########### INSERT ALLIANCE RANKING TO DB ############
    db_connection = sqlite3.connect(db_file)

    ## merge alliance rankings
    alliance_ranking, _ = extract_alliance_ranking_info(data, nation)

    ## merge ranking alliances with db alliances
    alliances = insert_alliances_to_db(db_connection, alliance_ranking, updateNames=False)

    ## insert alliance rankings to db
    alliance_data_collection_ids = insert_alliances_rankings_to_db(db_connection, alliance_ranking, alliances,
                                                                   timestamp)
    db_connection.commit()
    db_connection.close()

    ########### INSERT COMMANDER RANKING TO DB ############

    ## merge commander rankings
    commander_ranking, diags = extract_commander_ranking_info(data, nation)
    gui.init_ranking_data({"commander_ranking": commander_ranking, "diagnostic": diags})

    yield 1

    ## merge commander rankings with db commanders
    db_connection = sqlite3.connect(db_file)
    commanders, matched_commanders_to_insert, commanders_to_insert, stats, commander_to_insert_scores, commander_db_scores, non_matched_commanders_db = match_commanders_to_db(
        db_connection, commander_ranking, alliances, timestamp, detailed_alliances)
    db_connection.commit()
    db_connection.close()
    gui.init_fusion_data({
        "matched_commanders_to_insert": matched_commanders_to_insert,
        "commanders_to_insert": commanders_to_insert,
        "matching_stats": stats,
        "commander_to_insert_scores": commander_to_insert_scores,
        "commander_db_scores": commander_db_scores,
        "non_matched_commanders_db": non_matched_commanders_db})
    yield 2

    db_connection = sqlite3.connect(db_file)
    ## insert commanders to db
    commanders = insert_new_commanders_and_update_matched(db_connection, commanders, commander_ranking, alliances,
                                                          detailed_alliances, timestamp, matched_commanders_to_insert,
                                                          commanders_to_insert, non_matched_commanders_db)

    ## insert commander rankings to db
    commander_data_collection_ids = insert_commanders_rankings_to_db(db_connection, commander_ranking, commanders,
                                                                     timestamp)

    ## insert strongest commander event rankings to db
    if "event_type" in data:
        insert_sc_to_db(db_connection, data, data["event_type"], alliance_data_collection_ids,
                        commander_data_collection_ids,
                        timestamp)

    db_connection.commit()
    db_connection.close()
    gui.finalize_data_insertion()
    yield 3





