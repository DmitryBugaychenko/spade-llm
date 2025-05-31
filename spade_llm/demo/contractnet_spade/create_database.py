import sqlite3
import sys

import numpy as np
import pandas


def main():
    data_path = sys.argv[1]
    print("Looking for database at " + data_path)

    print("Reading data...")
    df = pandas.read_csv(
        data_path + "/transactions_data.csv",
        usecols={"client_id","date","amount","mcc"},
        dtype={"client_id":np.uint32, "mcc":np.uint16},
        converters={
            "amount" : lambda s: float(s[1:]),
            "date": lambda s: s[:10]})
    df['date'] = df['date'].astype("string")

    df.info()
    print(df.head(10))

    print("Pivoting data...")
    pivot_df = df.pivot_table(
        index="client_id",
        columns="mcc",
        values="amount",
        aggfunc="sum"
    ).fillna(0)

    pivot_df["client_id"] = pivot_df.index

    pivot_df.info()
    print(pivot_df.head(10))

    db = sqlite3.connect(data_path + "/sqlite.db")

    print("Importing data to SQL Lite...")
    df.to_sql(
        "transactions", db, if_exists='replace', index=False)
    pivot_df.to_sql(
        "spendings", db, if_exists='replace', index=False)

    print("Creating indices...")
    db.execute("CREATE INDEX date_ind ON transactions (date)")
    db.execute("CREATE INDEX date_mcc_ind ON transactions (date, mcc)")

    print("Following tables created:")
    cursor = db.execute("SELECT sql FROM sqlite_master WHERE name=?;", ["transactions"])
    sql = cursor.fetchone()[0]
    cursor.close()
    print(sql)

    cursor = db.execute("SELECT sql FROM sqlite_master WHERE name=?;", ["spendings"])
    sql = cursor.fetchone()[0]
    cursor.close()
    print(sql)

    print("Flushing data...")
    db.commit()
    db.close()
    print("DONE")

if __name__ == "__main__":
    main()