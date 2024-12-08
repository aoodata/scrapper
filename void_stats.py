import sqlite3
from utils import get_date_of_first_strongest_commander_event_before, collection_type_id, get_collection_id
import pandas as pd
import numpy as np

query = """select commanders.canonical_name as "Name",
       alliances.name_short as "Alliance",
       data_loss1.value - data_loss2.value as "Loss",
         data_kill1.value - data_kill2.value as "Kill",
         data_power1.value - data_power2.value as "Power",
            data_reputation1.value - data_reputation2.value as "Reputation",
            data_ke_score.value as "KE Score",
            data_ke_score.rank as "KE Rank",
            data_sc_score.value as "SC Score",
            data_sc_score.rank as "SC Rank"
from commander_ranking_data as data_loss1, commander_ranking_data as data_loss2, commanders, alliances,
     commander_ranking_data as data_kill1, commander_ranking_data as data_kill2,
        commander_ranking_data as data_power1, commander_ranking_data as data_power2,
        commander_ranking_data as data_reputation1, commander_ranking_data as data_reputation2
left outer join (select * from commander_ranking_data as tmp1 where tmp1.data_collection_id = ?) as data_ke_score on data_loss1.commander_id = data_ke_score.commander_id
left outer join (select * from commander_ranking_data as tmp2 where tmp2.data_collection_id = ?) as data_sc_score on data_loss1.commander_id = data_sc_score.commander_id
         where data_loss1.data_collection_id = ? and data_loss2.data_collection_id = ?
            and data_reputation1.data_collection_id = ? and data_reputation2.data_collection_id = ?
            and data_kill1.data_collection_id = ? and data_kill2.data_collection_id = ?
            and data_power1.data_collection_id = ? and data_power2.data_collection_id = ?
              and data_loss1.commander_id = data_reputation1.commander_id and data_loss1.commander_id = data_reputation2.commander_id
                and data_loss1.commander_id = data_power1.commander_id and data_loss1.commander_id = data_power2.commander_id
                  and data_loss1.commander_id = data_kill1.commander_id and data_loss1.commander_id = data_kill2.commander_id
           and data_loss1.commander_id = data_loss2.commander_id
           and data_loss1.commander_id = commanders.id
           and commanders.alliance_id = alliances.id;
"""





def extract_void_stats(dbFile, void_date):
    import time
    conn = sqlite3.connect(dbFile)
    c = conn.cursor()
    #convert date to timestamp
    void_date = time.mktime(void_date.timetuple())

    c.execute(query,
        (get_collection_id(c, "commander_ke_void", void_date, False),
         get_collection_id(c, "commander_sc_void", void_date, False),
         get_collection_id(c, "commander_loss", void_date, False),
         get_collection_id(c, "commander_loss", void_date, True),
         get_collection_id(c, "commander_reputation", void_date, False),
         get_collection_id(c, "commander_reputation", void_date, True),
         get_collection_id(c, "commander_kill", void_date, False),
         get_collection_id(c, "commander_kill", void_date, True),
         get_collection_id(c, "commander_power", void_date, False),
         get_collection_id(c, "commander_power", void_date, True)
         )
    )


    result = c.fetchall()
    conn.close()

    result = pd.DataFrame(result,
                          columns=["Name", "Alliance", "Loss", "Kill", "Power", "Reputation", "KE Score", "KE Rank",
                                   "SC Score", "SC Rank"])

    # set columns type to int except for Name and Alliance
    result = result.astype(
        {"Loss": 'Int64', "Kill": 'Int64', "Power": 'Int64', "Reputation": 'Int64', "KE Score": 'Int64',
         "KE Rank": 'Int64', "SC Score": 'Int64', "SC Rank": 'Int64'})

    # add column kill/loss ratio after column kill
    result.insert(4, "Kill/Loss", np.round((result["Kill"] + 0.0001)/ result["Loss"], 2))
    # if loss is 0, replace Kill/Loss with 0
    result["Kill/Loss"] = result["Kill/Loss"].replace(np.inf, np.nan)
    # replace nan with 0
    #result["Kill/Loss"] = result["Kill/Loss"].fillna(0)
    # add column kill minus loss after column Kill/Loss
    result.insert(5, "Kill-Loss", result["Kill"] - result["Loss"])
    # if name starts with an = sign, add a ` to the start of the name
    result["Name"] = result["Name"].apply(lambda x: "'" + x if x.startswith("=") else x)

    # sort by ke rank, then sc rank, then kill/loss ratio, then kill-loss
    result = result.sort_values(by=["KE Rank", "SC Rank", "Kill-Loss", "Kill/Loss"],
                                ascending=[True, True, False, False])
    return result

if __name__ == "__main__":
    date = get_date_of_first_strongest_commander_event_before(event_type="void")["date"]
    dbFile = "aoodata.github.io/data/385DB.sqlite"
    result = extract_void_stats(dbFile, date)

    tsv = result.to_csv(index=False, sep="\t", encoding="utf-8", header=False)
    # send tsv to clipboard
    import pyperclip
    pyperclip.copy(tsv)
    print(result)