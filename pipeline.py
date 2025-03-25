import pandas as pd
import numpy as np

from typing import Dict, Tuple

from utils import (
    find_dead_birds,
    stop_num_count,
    calculate_stops_duration,
)
from geopy.distance import geodesic
import logging
import warnings
import time

warnings.filterwarnings("ignore")

log_file = "pipeline_log.txt"
logging.basicConfig(filename=log_file, encoding='utf-8')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


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
    # group.dropna(inplace=True, subset=['country_next'])

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


def calc_time_in_rus(group: pd.DataFrame, debug: bool = False) -> Tuple[pd.Timedelta | None]:
    border_cross_data = border_cross(group, debug=debug)

    time_in_rus: pd.Timedelta = pd.Timedelta(0)
    stops_in_hunt: pd.Timedelta = pd.Timedelta(0)
    stops_wo_hunt: pd.Timedelta = pd.Timedelta(0)

    for i, j in zip(border_cross_data["In"], border_cross_data["Out"]):

        if (group.loc[group["timestamp"] == i].stops.values is True) and (
            group.loc[group["timestamp"] == j].stops.values is True
        ):
            if group.loc[group["timestamp"] == i].hunting.values is True:
                stops_in_hunt += j - i
            else:
                stops_wo_hunt += j - i

        time_in_rus += j - i
        if debug:
            print(
                j - i,
                group.loc[group["timestamp"] == i].stops.values,
                group.loc[group["timestamp"] == j].stops.values,
            )

    return time_in_rus, stops_in_hunt, stops_wo_hunt


# Function to calculate the distance between two geographical points
def calculate_distance(coord1, coord2) -> float:
    return geodesic(coord1, coord2).meters


# Function to calculate the first day of nesting
def calculate_nest_start(df: pd.DataFrame, max_dist: int = 50, debug: bool = False) -> pd.DataFrame:
    # Filter the data for records after June 1th of each year
    df = df[df["timestamp"] > pd.to_datetime(df["year"].astype(str) + "-06-01")]

    # Group by each unique bird-year combination
    groups = df.groupby(["Bird_id", "year"])

    nest_start_data = []

    for (bird_id, year), group in groups:
        group = group.sort_values(by="timestamp")
        central_point = None

        if debug:
            print(bird_id, year)

        # for date, day_group in group.groupby(group['timestamp'].dt.date):
        for date, day_group in group.groupby(pd.Grouper(key="timestamp", freq="12h")):
            # Calculate the daily distances from the central point
            distances = day_group.apply(
                lambda row: (
                    calculate_distance((row["location_lat"], row["location_long"]), central_point)
                    if central_point
                    else 0
                ),
                axis=1,
            )

            # If there is no central point yet, initialize it
            if (central_point is None) and (day_group["location_lat"].iloc[0] >= 68):
                central_point = (
                    day_group["location_lat"].iloc[0],
                    day_group["location_long"].iloc[0],
                )
                continue

            # Check if the bird stays within 'max_dist' meters from the central point for the entire day
            try:
                if distances.max() <= max_dist:
                    nest_start_data.append(
                        {
                            "Bird_id": bird_id,
                            "year": year,
                            "nest_start_date": date.date(),
                            "nest": central_point,
                        }
                    )
                    break
                else:
                    # Update the central point as the median point of the day's locations
                    central_point = (
                        day_group["location_lat"].median(),
                        day_group["location_long"].median(),
                    )
            except ValueError:
                break  # if error in data - pass the bird

    return pd.DataFrame(nest_start_data)


def main(file_name: str) -> None:
    logging.info("Start of the pipeline")
    # get data
    birds_all = pd.read_parquet(file_name)

    # transform and clean of the data
    # birds_all = find_dead_birds(birds_all)
    birds_all["timestamp"] = pd.to_datetime(birds_all["timestamp"])
    birds_all = birds_all[
        (birds_all["timestamp"].dt.year <= 2023) & (birds_all["timestamp"].dt.year >= 2016)
    ]

    # select only spring migration
    birds_all = birds_all[birds_all['timestamp'] < pd.to_datetime(birds_all['year'].astype(str) + '-07-31')]

    birds_in_ru = birds_all.groupby(["Bird_id", "year"]).agg(
        in_rus=("country", lambda x: "Россия" in x.values)
    )

    birds_to_delete = birds_in_ru.loc[birds_in_ru["in_rus"] == False].index
    # Set 'Bird_id' and 'year' as the index of the DataFrame
    birds_all = birds_all.set_index(["Bird_id", "year"])

    # Filter out the rows with the given pairs
    birds_all = birds_all[~birds_all.index.isin(birds_to_delete)]
    logging.info(f"Remove birds without Russia. Number of birds: {birds_all.shape[0]}")

    # delete birds that dont reach polar circle
    last_lat_for_bird = birds_all.groupby(["Bird_id", "year"]).agg(
        last_lat=("location_lat", 'max')
    )
    killed_birds = last_lat_for_bird.loc[last_lat_for_bird['last_lat'] <= 66.562].index
    birds_all = birds_all[~birds_all.index.isin(killed_birds)]
    logging.info(f"Remove killed birds. Number of birds: {birds_all.shape[0]}")

    # Reset index if needed
    birds_all.reset_index(inplace=True)

    logging.info("File read and transformed!")

    # get last cross date of the border with Russia
    last_cross_date = (
        birds_all.groupby(["Bird_id", "year"]).apply(get_last_date_of_border_cross).reset_index()
    )

    last_cross_date.columns = ["Bird_id", "year", "cross_date"]

    # save the lsat crossing date file
    last_cross_date.to_csv(f".\\data\\{timestr}-last_cross_date_to_rus.csv")
    logging.info("File last_cross_date saved!")
    logging.info(f'Number of rows: {last_cross_date.shape[0]}')

    # calculate the time bird spent in Russia
    time_in_ru = birds_all.groupby(["Bird_id", "year"]).apply(calc_time_in_rus).reset_index()
    time_in_ru.columns = ["Bird_id", "year", "time_in_ru"]
    time_in_ru[["time_in_ru", "stops_in_hunt", "stops_wo_hunt"]] = pd.DataFrame(
        time_in_ru["time_in_ru"].tolist(), index=time_in_ru.index
    )
    time_in_ru.to_csv(f".\\data\\{timestr}-time_in_rus.csv")
    logging.info("File time_in_ru saved!")
    logging.info(f'Number of rows: {time_in_ru.shape[0]}')

    # add crossing date and time in Russia to the main dataframe
    birds_all = pd.merge(birds_all, time_in_ru, on=["Bird_id", "year"])
    birds_all = pd.merge(birds_all, last_cross_date, on=["Bird_id", "year"])

    # calculation of stops and stops duration after the last border cross
    birds = birds_all.loc[
        (birds_all["country"] == "Россия")
        & (
            (birds_all["date"].dt.month < 6)
            | ((birds_all["date"].dt.month == 6) & (birds_all["date"].dt.day < 10))
        )
    ]

    columns = ["cross_date", "date"]
    birds[columns] = birds[columns].apply(pd.to_datetime, errors="coerce")
    birds = (
        birds.groupby(["Bird_id", "year"])
        .apply(lambda x: x[x["date"] > x["cross_date"]])
        .reset_index(drop=True)
    )

    stop_number = stop_num_count(birds, ["Species", "Bird_id", "hunting", "year"])
    stop_duration = calculate_stops_duration(birds)

    merged_df = pd.merge(
        stop_number[["Bird_id", "year", "hunting", "stops_number"]],
        stop_duration[["Bird_id", "year", "Mean stopover duration, days", "hunting"]],
        on=["Bird_id", "year", "hunting"],
        how="inner",
    )

    result_stops_hunting = (
        merged_df.groupby(["Bird_id", "year", "hunting"])
        .agg(
            stops_with_hunting=("stops_number", "size"),  # Count number of stops
            sum_stopover_duration=(
                "Mean stopover duration, days",
                "sum",
            ),  # Sum of stops duration
        )
        .reset_index()
    )

    # Reshape the result to have separate columns for hunting=True and hunting=False
    result_pivot = result_stops_hunting.pivot_table(
        index=["Bird_id", "year"],
        columns="hunting",
        values=["stops_with_hunting", "sum_stopover_duration"],
        fill_value=0,
    )

    # Flatten the multi-index columns
    result_pivot.columns = ["_".join([str(col[0]), str(col[1])]) for col in result_pivot.columns]
    result_pivot = result_pivot.reset_index()

    result_pivot.to_csv(f".\\data\\{timestr}-stops_hunting.csv")
    logging.info("File stops_hunting saved!")
    logging.info(f'Number of rows: {result_pivot.shape[0]}')

    # find nests
    nest_time_all = calculate_nest_start(birds_all, max_dist=50)

    # combine all data
    result = pd.merge(last_cross_date, time_in_ru, on=["Bird_id", "year"], how="left")
    result = pd.merge(result, result_pivot, on=["Bird_id", "year"], how="left")
    result = pd.merge(result, nest_time_all, on=["Bird_id", "year"], how="left")

    columns = ["cross_date", "nest_start_date"]
    result[columns] = result[columns].apply(pd.to_datetime, errors="coerce")
    result["time_in_ru"] = pd.to_timedelta(result["time_in_ru"])

    result["time_in_ru_before_nest"] = result["nest_start_date"] - result["cross_date"]
    result["time_full_before_nest"] = result["time_in_ru"] + result["time_in_ru_before_nest"]

    result["nest"] = result["nest"].fillna("No nest")

    dates = []

    for row in result.iterrows():
        year = row[1][1]

        dates.append(row[1][3] + (np.datetime64(f"{year}-06-10") - row[1][2]))

    result["time_before_fix"] = dates

    result["sum_stopover_duration_False"] = pd.to_timedelta(result["sum_stopover_duration_False"], unit="D")
    result["sum_stopover_duration_True"] = pd.to_timedelta(result["sum_stopover_duration_True"], unit="D")
    result["stops_in_hunt"] = pd.to_timedelta(result["stops_in_hunt"])
    result["stops_wo_hunt"] = pd.to_timedelta(result["stops_wo_hunt"])

    result["stops_dur_in_hunt"] = result["stops_in_hunt"] + result["sum_stopover_duration_True"]
    result["stops_dur_wo_hunt"] = result["stops_wo_hunt"] + result["sum_stopover_duration_False"]

    result["check"] = result["time_before_fix"] - result["stops_dur_in_hunt"] - result["stops_dur_wo_hunt"]

    result = result.drop(
        [
            "stops_in_hunt",
            "stops_wo_hunt",
            "sum_stopover_duration_True",
            "sum_stopover_duration_False",
        ],
        axis=1,
    )

    result.to_csv(f".\\data\\{timestr}-combined_data.csv")

    logging.info(f'Number of rows: {result.shape[0]}')
    logging.info("Final file saved!")
    logging.info('Files are ready, now converting the dates')

    convert_dates(f".\\data\\{timestr}-combined_data.csv", save_file=True)

    logging.info("The end! Every thing is OK")


def timedelta_to_days(timedelta: pd.Timedelta) -> float:
    return timedelta.days + timedelta.seconds / 24 / 60 / 60


def days_from_start_of_year(dt: pd.Timestamp) -> float:
    # Start of the year for the given date
    try:
        start_of_year: pd.Timestamp = pd.Timestamp(year=dt.year, month=1, day=1)
    except TypeError:
        return 0

    # Calculate the difference between the date and the start of the year
    delta = dt - start_of_year

    # Return the difference in days as a float, including the decimal part for time
    return delta.days + delta.seconds / (24 * 60 * 60)


def convert_dates(file_name: str, save_file: bool) -> pd.DataFrame | None:
    df = pd.read_csv(file_name, index_col=0)

    df["time_in_ru"] = pd.to_timedelta(df["time_in_ru"]).apply(timedelta_to_days)
    df["time_in_ru_before_nest"] = pd.to_timedelta(df["time_in_ru_before_nest"]).apply(timedelta_to_days)
    df["time_full_before_nest"] = pd.to_timedelta(df["time_full_before_nest"]).apply(timedelta_to_days)
    df["time_before_fix"] = pd.to_timedelta(df["time_before_fix"]).apply(timedelta_to_days)
    df["stops_dur_in_hunt"] = pd.to_timedelta(df["stops_dur_in_hunt"]).apply(timedelta_to_days)
    df["stops_dur_wo_hunt"] = pd.to_timedelta(df["stops_dur_wo_hunt"]).apply(timedelta_to_days)

    df["cross_date"] = pd.to_datetime(df["cross_date"])
    df["nest_start_date"] = pd.to_datetime(df["nest_start_date"])

    df["cross_date_in_jd"] = df["cross_date"].apply(days_from_start_of_year)
    df["nest_start_date_jd"] = df["nest_start_date"].apply(days_from_start_of_year)

    if save_file:
        df.to_csv(f".\\data\\{timestr}-combined_data_final.csv")
        return None
    else:
        return df


if __name__ == "__main__":
    timestr = time.strftime("%Y%m%d-%H%M")
    file_path_raw = "data\\raw\\geese_all_10_2024_fix.parquet"

    main(file_path_raw)
