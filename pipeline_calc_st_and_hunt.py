import pandas as pd
from typing import Dict
from pathlib import Path
from icecream import ic


def stop_and_hunt(df: pd.DataFrame) -> Dict[str, float]:
    df['situation'] = df['stops'].astype(str) + '-' + df['hunting'].astype(str)

    # Step 1: Identify changes in situation
    df['situation_change'] = (df['situation'] != df['situation'].shift()).cumsum()

    # Step 2: Calculate start and end timestamps for each situation
    grouped = df.groupby('situation_change').agg(
        start_time=('timestamp', 'first'),
        end_time=('timestamp', 'last'),
        situation=('situation', 'first'),
        countries=('country', lambda x: set(x.dropna()))
    )

    # Add label based on country field
    grouped['single_country'] = grouped['countries'].apply(lambda x: len(x) == 1 and 'Россия' in x)

    # Step 3: Calculate duration for each situation
    grouped['duration'] = (grouped['end_time'] - grouped['start_time']).dt.total_seconds()

    # Step 4: Remove groups that do not meet the criteria
    # remove groups with duration less than 2 days
    grouped = grouped[grouped['duration'] > 172800]
    # remove groups with end_time after 10th of June
    grouped = grouped[grouped['end_time'] < pd.Timestamp('2024-06-10')]
    # remove groups with single_country False
    grouped = grouped[grouped['single_country']]

    # Calculate total durations for each unique situation
    situation_durations = grouped.groupby('situation')['duration'].sum()

    # Step 5: Organize into the desired result structure
    results = {
        'duration_stops_hunting': situation_durations.get('True-True', 0) / 86400,
        'duration_stops_no_hunting': situation_durations.get('True-False', 0) / 86400,
        'duration_no_stops_hunting': situation_durations.get('False-True', 0) / 86400,
        'duration_no_stops_no_hunting': situation_durations.get('False-False', 0) / 86400,
    }

    return results


def clean_data_raw(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    df_raw - dataframe from parquet file made from the raw data

    Clean data according to the requirements:
    - Transform 'timestamp' to datetime
    - Filter only data for the period from 2016 to 2023
    - Filter only data for spring migration
    - Remove birds that do not reach polar circle (latitude <= 66.562)
 
    """
    # transform and clean of the data
    df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"])
    df_raw = df_raw[
        (df_raw["timestamp"].dt.year <= 2023) & (df_raw["timestamp"].dt.year >= 2016)
    ]

    # select only spring migration
    df_raw = df_raw[
        df_raw["timestamp"] < pd.to_datetime(df_raw["year"].astype(str) + "-07-31")
    ]

    birds_in_ru = df_raw.groupby(["Bird_id", "year"]).agg(
        in_rus=("country", lambda x: "Россия" in x.values)
    )

    birds_to_delete = birds_in_ru.loc[birds_in_ru["in_rus"] == False].index
    # Set 'Bird_id' and 'year' as the index of the DataFrame
    df_raw = df_raw.set_index(["Bird_id", "year"])

    # Filter out the rows with the given pairs
    df_raw = df_raw[~df_raw.index.isin(birds_to_delete)]

    # delete birds that dont reach polar circle
    last_lat_for_bird = df_raw.groupby(["Bird_id", "year"]).agg(
        last_lat=("location_lat", "max")
    )
    killed_birds = last_lat_for_bird.loc[last_lat_for_bird["last_lat"] <= 66.562].index
    df_raw = df_raw[~df_raw.index.isin(killed_birds)]

    # Reset index if needed
    df_raw.reset_index(inplace=True)

    return df_raw

def clean_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Use after clean_data_raw and adding last_date_of_border_cross and first_date_of_border_cross

    - Filter only data for Russia
    - Filter only data for the period from 1st of January to 10th of June
    - Filter only data where timestamp is greater or equal to cross_date (now not used)
    """

    df = df_raw.loc[
        (df_raw["country"] == "Россия"),
        [
            "timestamp",
            "stops",
            "Bird_id",
            "year",
            "last_date_of_border_cross",
            "first_date_of_border_cross",
            "hunting",
            "location_long",
            "location_lat",
            "country"
        ],
    ]
    # df = df[df["timestamp"] >= df["cross_date"]]
    df = df[
        (df["timestamp"].dt.month < 6)
        | ((df["timestamp"].dt.month == 6) & (df["timestamp"].dt.day < 10))
    ]

    return df


def border_cross(group: pd.DataFrame, debug: bool = False) -> Dict | None:

    group = group.copy()
    group = group[group.country != "None"]
    bird_name = group["Bird_id"].values[0]
    year = group["year"].values[0]
    group.dropna(inplace=True, subset=["country"])

    if debug:
        print(bird_name, year)

    if "Россия" not in group["country"].values:
        return {"Bird_name": bird_name, "In": [], "Out": []}

    group["country_prev"] = group["country"].shift(1)
    group["country_next"] = group["country"].shift(-1)

    group.dropna(inplace=True, subset=["country_prev", "country_next"])

    if group.shape[0] == 0:
        return {"Bird_name": bird_name, "In": [], "Out": []}

    rus_in = group.loc[(group["country"] == "Россия") & (group["country_prev"] != "Россия")]["timestamp"]
    rus_out = group.loc[(group["country"] != "Россия") & (group["country_prev"] == "Россия")]["timestamp"]

    return {"Bird_name": bird_name, "In": rus_in.to_list(), "Out": rus_out.to_list()}


def get_last_date_of_border_cross(group: pd.DataFrame, debug: bool = False) -> pd.Timestamp | None:
    border_cross_data = border_cross(group, debug=debug)

    if len(border_cross_data["In"]) == 0:
        return None

    return border_cross_data["In"][-1]

def get_first_date_of_border_cross(group: pd.DataFrame, debug: bool = False) -> pd.Timestamp | None:
    border_cross_data = border_cross(group, debug=debug)

    if len(border_cross_data["In"]) == 0:
        return None

    return border_cross_data["In"][0]


def main():
    # Load data
    file_path = Path(".\\data\\raw\\geese_all_10_2024_fix.parquet")
    df_raw = pd.read_parquet(file_path)

    # Clean data
    df = clean_data_raw(df_raw)

    # add last date and first date of border cross
    last_cross_date = df.groupby(["Bird_id", "year"]).apply(get_last_date_of_border_cross).reset_index()
    last_cross_date.columns = ["Bird_id", "year", "last_date_of_border_cross"]
    df = pd.merge(df, last_cross_date, on=["Bird_id", "year"])

    first_cross_date = df.groupby(["Bird_id", "year"]).apply(get_first_date_of_border_cross).reset_index()
    first_cross_date.columns = ["Bird_id", "year", "first_date_of_border_cross"]
    df = pd.merge(df, first_cross_date, on=["Bird_id", "year"])

    # Remove unnecessary columns
    df = clean_data(df)

    # Filter only data where timestamp is greater or equal to last_date_of_border_cross
    df = df[df["timestamp"] >= df["last_date_of_border_cross"]]
    # df = df[df["timestamp"] >= df["first_date_of_border_cross"]]


    # Calculate stop and hunt
    stops_df = df.groupby(["Bird_id", "year"]).apply(stop_and_hunt)
    stops_df = stops_df.apply(pd.Series)
    stops_df = stops_df.reset_index()

    # add last date and first date of border cross
    stops_df = pd.merge(stops_df, last_cross_date, on=["Bird_id", "year"])
    stops_df = pd.merge(stops_df, first_cross_date, on=["Bird_id", "year"])

    # save results in data folder with name "{current data and time} + results.csv"
    stops_df.to_csv(Path(f".\\data\\{pd.Timestamp.now().strftime('%Y-%m-%d %H-%M-%S')}_results.csv"), index=False)


if __name__ == '__main__':
    main()