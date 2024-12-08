import sqlite3
import datetime
import time



def clean_db(dbFile, months):
    """
    Remove all commanders that have not been updated in the last months months and compact the database
    :param dbFile: sqlite database file
    :param months: number of months
    :return: nothing
    """
    date = datetime.date.today()
    date = date - datetime.timedelta(days=months*30)
    timestamp = time.mktime(date.timetuple())

    conn = sqlite3.connect(dbFile)
    c = conn.cursor()

    # find all commanders whose last data collection is older than timestamp
    c.execute(
        """select commanders.id, commanders.canonical_name 
                from commanders, data_collections, main.commander_ranking_data 
                where commanders.id = commander_ranking_data.commander_id 
                    and data_collections.id = commander_ranking_data.data_collection_id 
                GROUP BY commanders.id
                HAVING max(data_collections.date) < ?""", (timestamp,))

    commanders = c.fetchall()
    for commander in commanders:
        commander_id = commander[0]
        c.execute("delete from commander_ranking_data where commander_id = ?", (commander_id,))
        c.execute("delete from commander_names where commander_id = ?", (commander_id,))
        c.execute("delete from commanders where id = ?", (commander_id,))

    conn.commit()
    conn.execute("VACUUM")
    conn.close()

if __name__ == "__main__":
    filename = "aoodata.github.io/data/385TestDB.sqlite"
    #size of db
    import os
    print(os.path.getsize(filename))

    clean_db(filename, 6)

    print(os.path.getsize(filename))