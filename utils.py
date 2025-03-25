import os
import re
from itertools import product, tee, islice
from glob import glob
# from difflib import SequenceMatcher
import numpy as np
import pandas as pd
import dask.dataframe as dd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from statannotations.Annotator import Annotator
# from scipy.stats import kstest, spearmanr
# from sklearn.metrics import DistanceMetric
# from statannot import add_stat_annotation

###


def csv_to_xlsx(csv):
    """
    Преобразует файл формата CSV в файл формата Excel (XLSX).

    Параметры:
    - csv (str): Путь к файлу формата CSV.

    Возвращает:
    Нет возвращаемого значения.

    Пример использования:
    csv_to_xlsx("path/to/file/data.csv")
    # Создает файл "data.xlsx" с данными из файла "data.csv".

    """
    df = pd.DataFrame()
    out = csv.split(".")[0] + ".xlsx"
    df = pd.read_csv(csv, sep=",", decimal=",", engine="python")
    df.to_excel(out, float_format="%.4f")


def csv_to_xlsx_GDD(csv):
    """
    Преобразует файл формата CSV в файл формата Excel (XLSX), выполняя определенные преобразования данных.

    Параметры:
    - csv (str): Путь к файлу формата CSV.

    Возвращает:
    Нет возвращаемого значения.

    Пример использования:
    csv_to_xlsx_GDD("path/to/file/data.csv")
    # Создает файл "data.xlsx" с преобразованными данными из файла "data.csv".

    """

    df = pd.DataFrame()
    out = csv.split(".")[0] + ".xlsx"
    df = pd.read_csv(csv, sep=",", decimal=",", engine="python")

    df["duration, d"] = df["duration"].str.replace(" days", "").astype(int)
    df["gdd_diff, d"] = df["gdd_diff"].str.replace(" days", "").astype(int)
    df.loc[:, ["point_gdd", "bird_arrival", "bird_leave"]] = df.loc[
        :, ["point_gdd", "bird_arrival", "bird_leave"]
    ].astype("datetime64[M]")
    df["day_point_gdd"] = df["point_gdd"].dt.dayofyear
    df["day_bird_arrival"] = df["bird_arrival"].dt.dayofyear
    df["day_bird_leave"] = df["bird_leave"].dt.dayofyear
    df.to_excel(out, float_format="%.4f")


def parquet_to_xlsx(parq):
    """
    Преобразует файл формата Parquet в файл формата Excel (XLSX).

    Параметры:
    - parq (str): Путь к файлу формата Parquet.

    Возвращает:
    Нет возвращаемого значения.

    Пример использования:
    parquet_to_xlsx("path/to/file/data.parquet")
    # Создает файл "data.xlsx" с данными из файла "data.parquet".

    """
    df = pd.DataFrame()
    out = parq.split(".")[0] + ".xlsx"
    df = pd.read_parquet(parq, engine="fastparquet")
    df.to_excel(out, float_format="%.4f")


# ADD FEATURES


def name_family_sp(filepath):
    """
    Определяет вид (вид, семейство), название птицы и проект на основе пути к файлу.

    Параметры:
    - filepath (str): Путь к файлу.

    Возвращает:
    - sp (str): Вид птицы.
    - family (str): Семейство птицы.
    - bird_name (str): Название птицы.
    - project (str): Проект.

    Пример использования:
    sp, family, bird_name, project = name_family_sp(
        "path\\to\\file\\Sp_Sparrow\\Families\\Some_Bird")
    # Вывод: Sp_Sparrow, Family_Some_Family, ID_Bird, Project Families

    """

    family = None
    if "семьи" in filepath.split("\\")[-3]:  # families
        sp = filepath.split("\\")[-4]
        family = filepath.split("\\")[-2]
    elif "Андре" in filepath.split("\\")[-3]:  # from andrew
        sp = filepath.split("\\")[-4]
    else:
        sp = filepath.split("\\")[-3]
    bird_name = (
        filepath.split("\\")[-1].split(".")[0].replace("_", "^", 1).split("^")[-1]
    )
    project = filepath.split("\\")[-2]
    print(f"Sp_{sp}, Family_{family}, ID_{bird_name}, Project {project}")
    return sp, family, bird_name, project


def add_age_sex(df, age_sex_path):
    """
    Добавляет информацию о возрасте и поле птицы из файла Excel к исходному DataFrame.

    Параметры:
    - df (pandas DataFrame): Исходный DataFrame, к которому будет добавлена информация.
    - age_sex_path (str): Путь к файлу Excel с информацией о возрасте и поле птицы.

    Возвращает:
    - df (pandas DataFrame): DataFrame, в который добавлена информация о возрасте и поле птицы.

    Пример использования:
    data = pd.DataFrame({'Bird_id': [1, 2, 3],
                         'Value': [10, 20, 30]})
    age_sex_file = "path/to/file/age_sex.xlsx"
    data_with_info = add_age_sex(data, age_sex_file)
    # Возвращает DataFrame с информацией о возрасте и поле птицы, объединенной с исходным DataFrame.

    """
    age_sex = pd.read_excel(age_sex_path)
    df = df.merge(age_sex, on="Bird_id", how="left")
    return df


def sex_age_clean(path):
    """
    Выполняет очистку данных о поле и возрасте птиц из файла Excel.

    Параметры:
    - path (str): Путь к файлу Excel.

    Возвращает:
    - df (pandas DataFrame): Очищенный DataFrame с данными о поле и возрасте птиц.

    Пример использования:
    data_file = "path/to/file/data.xlsx"
    cleaned_data = sex_age_clean(data_file)
    # Возвращает очищенный DataFrame с данными о поле и возрасте птиц из файла Excel.

    """
    df = (
        pd.concat(pd.read_excel(path, sheet_name=None))
        .reset_index()
        .loc[
            :,
            [
                "Animal id",
                "пол",
                "возраст при кольцевании",
                "возраст при кольцевании\n(отмечается только juv)",
            ],
        ]
    )
    df.loc[
        df["возраст при кольцевании\n(отмечается только juv)"].isna() != True,
        "возраст при кольцевании",
    ] = "juv"
    df["возраст при кольцевании"].fillna("adult", inplace=True)
    df = df.iloc[:, :-1]
    df["пол"].fillna("Unknown", inplace=True)
    df.rename(
        columns={
            "Animal id": "Bird_id",
            "возраст при кольцевании": "Age",
            "пол": "Sex",
        },
        inplace=True,
    )

    return df


def add_last_date(df):
    """
    Добавляет последнюю дату в РФ в DataFrame и выполняет сортировку по дате.

    Параметры:
    - df (pandas DataFrame): Исходный DataFrame.

    Возвращает:
    - out (pandas DataFrame): DataFrame с добавленной последней датой и отсортированным по дате.

    Пример использования:
    data = pd.DataFrame(...)
    updated_data = add_last_date(data)
    # Возвращает DataFrame с добавленной последней датой и отсортированным по дате.

    """
    last_row = df.iloc[-1].to_frame().T
    last_row["timestamp"] = pd.to_datetime(df["year"].astype(str) + "-06-01")
    print(last_row)
    out = (
        pd.concat([last_row, df], axis=0)
        .sort_values(by=["timestamp"], ascending=True)
        .reset_index()
    )
    return out.astype({"location_long": np.int64, "location_lat": np.int64})


# vectorized haversine function
# взято из  https://stackoverflow.com/questions/40452759/pandas-latitude-longitude-to-distance-between-successive-rows


def haversine(lat1, lon1, lat2, lon2, to_radians=True, earth_radius=6371):
    """
    slightly modified version: of http://stackoverflow.com/a/29546836/2901002

    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees or in radians)

    All (lat, lon) coordinates must have numeric dtypes and be of equal length.

    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    a = (
        np.sin((lat2 - lat1) / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2.0) ** 2
    )

    return earth_radius * 2 * np.arcsin(np.sqrt(a))


def add_distance_speed(df):
    """
    Добавляет расстояние и скорость в DataFrame на основе координат и времени.

    Параметры:
    - df (pandas DataFrame): Исходный DataFrame с данными.

    Возвращает:
    - df (pandas DataFrame): DataFrame с добавленными столбцами расстояния и скорости.

    Пример использования:
    data = pd.DataFrame(...)
    updated_data = add_distance_speed(data)
    # Возвращает DataFrame с добавленными столбцами расстояния и скорости.

    """
    for year in df["year"].unique():
        df_year = df.loc[df["year"] == year].copy()
        df.loc[df["year"] == year, "hours_in_raion"] = df_year[
            "timestamp"
        ].diff() / np.timedelta64(1, "h")
        df.loc[df["year"] == year, "dist, km"] = haversine(
            df_year["location_long"].shift(1).values,
            df_year["location_lat"].shift(1).values,
            df_year["location_long"].values,
            df_year["location_lat"].values,
        )
        df.loc[df["year"] == year, "V, km/h"] = df["dist, km"] / df["hours_in_raion"]
    return df


def create_dir(dir_path):
    os.makedirs(dir_path, exist_ok=True)


def merge_gdd(gdd_path):
    """
    Объединяет файлы GDD (Growing Degree Days) в один DataFrame.

    Параметры:
    - gdd_path (str): Путь к файлам GDD.

    Возвращает:
    - merged_gdd (pandas DataFrame): Объединенный DataFrame с данными GDD.

    Пример использования:
    gdd_directory = "path/to/gdd/files"
    merged_data = merge_gdd(gdd_directory)
    # Возвращает объединенный DataFrame с данными GDD.

    """
    merged_gdd = pd.DataFrame()
    pattern = r"(?:19[0-9][0-9]|20[0-2][0-9])"
    for filepath in glob(gdd_path):
        gdd_df = pd.read_excel(filepath)
        gdd_df["year"] = re.findall(pattern, filepath)[-1]
        merged_gdd = pd.concat([merged_gdd, gdd_df])
    merged_gdd = merged_gdd.loc[
        :,
        [
            "point",
            "point_lat",
            "point_lon",
            "point_gdd",
            "bird_name",
            "duration, d",
            "temp",
            "gdd_diff, d",
            "day_point_gdd",
            "day_bird_arrival",
            "day_bird_leave",
            "year",
        ],
    ].rename(columns={"bird_name": "Bird_id"})
    return merged_gdd


def measure_coords_by_day(input):
    """
    Измеряет координаты в начале и в конце дня, и возвращает агрегированный DataFrame.

    Параметры:
    - input (pandas DataFrame): Исходный DataFrame с данными.

    Возвращает:
    - coord (pandas DataFrame): Агрегированный DataFrame с измеренными координатами и расстояниями.

    Пример использования:
    data = pd.DataFrame(...)
    measured_data = measure_coords_by_day(data)
    # Возвращает агрегированный DataFrame с измеренными координатами и расстояниями.

    """
    input["date"] = input["timestamp"].dt.date
    coord = input.groupby(["year", "date"], as_index=False).agg(
        {"location_long": ["first", "last"], "location_lat": ["first", "last"]}
    )
    coord.columns = [
        "year",
        "date",
        "location_long_first",
        "location_long_last",
        "location_lat_first",
        "location_lat_last",
    ]
    coord["dist"] = haversine(
        coord["location_long_first"].values,
        coord["location_lat_first"].values,
        coord["location_long_last"].values,
        coord["location_lat_last"].values,
    )
    return coord


def stops_add(df):
    """
    Добавляет информацию о остановках на основе расстояний в DataFrame.
    Если за 2 дня пройдено меньше 30км, то считаем, что это остановка

    Параметры:
    - df (pandas DataFrame): Исходный DataFrame с данными.

    Возвращает:
    - out (pandas DataFrame): DataFrame с добавленной информацией о остановках.

    Пример использования:
    data = pd.DataFrame(...)
    data_with_stops = stops_add(data)
    # Возвращает DataFrame с добавленной информацией о остановках.

    """
    coord = measure_coords_by_day(df)
    stop_dist = (
        coord.groupby(["year", "date"])
        .agg({"dist": "sum"})
        .rolling(2)
        .agg({"dist": "sum"})
    )
    stop_dist.loc[stop_dist["dist"] <= 30, "stops"] = True
    # stop_dist['stops'].fillna(False)
    stop_dist.reset_index(inplace=True)
    stop_dist["prev_1"] = stop_dist["stops"].shift(-1)
    stop_dist.loc[(stop_dist["prev_1"] == True), "stops"] = True
    stop_dist["stops"].fillna(False, inplace=True)
    stop_dist.reset_index(inplace=True)
    out = df.merge(
        stop_dist.loc[:, ["date", "year", "stops"]], on=["year", "date"], how="left"
    )
    out["date"] = pd.to_datetime(out["date"])
    return out


def add_features(path, format="parquet"):
    """
    Обрабатывает файлы данных, добавляет дополнительные функции и сохраняет результат в указанном формате.

    Параметры:
    - path (str): Путь к файлам данных.
    - format (str, по умолчанию 'parquet'): Формат, в котором сохраняются результаты. Допустимые значения: 'xlsx', 'csv', 'parquet'.

    Возвращает:
    - None

    Пример использования:
    data_path = "path/to/data/files"
    add_features(data_path, format='parquet')
    # Обрабатывает файлы данных, добавляет дополнительные функции и сохраняет результат в формате Parquet.

    """
    for file in glob(path, age_sex_path, recursive=True):
        df = pd.read_csv(file, low_memory=False, parse_dates=["timestamp"])
        df["Species"], df["Family"], df["Bird_id"], df["project"] = name_family_sp(file)
        df["project"] = file.split("\\")[-2]
        # print(file.split('\\')[-2])
        df["year"] = df["timestamp"].dt.year
        df["month"] = df["timestamp"].dt.month
        df = add_age_sex(df, age_sex_path)
        df = stops_add(df)
        df.loc[df["Sex"].isnull(), "Sex"] = "Unknown"
        df = add_distance_speed(df)
        # save_path = f"features_parquet_test/{df['Species'].unique()[0]}/Family_{df['Family'].unique()[0]}/"
        if "family" in df["project"].unique()[0]:
            save_path = f"features_parquet_hunt_stops/{df['Species'].unique()[0]}/семьи/{df['project'].unique()[0]}/"
        else:
            save_path = f"features_parquet_hunt_stops/{df['Species'].unique()[0]}/{df['project'].unique()[0]}/"
        create_dir(save_path)
        ddf = dd.from_pandas(df, npartitions=1)
        if format == "xlsx":
            ddf.compute().to_excel(f"{save_path}/{df['Bird_id'].unique()[0]}.xlsx")
        elif format == "csv":
            ddf.to_csv(
                f"{save_path}/{df['Bird_id'].unique()[0]}.csv",
                single_file=True,
                compute=True,
            )
        elif format == "parquet":
            ddf.compute().to_parquet(f"{save_path}/{df['Bird_id'].unique()[0]}.parquet")
        else:
            print("Wrong format choose one: xlsx, csv, parquet")


def concat_all_files_spring(path, save_path, exclude_list):
    """
    Объединяет все файлы данных весеннего периода и сохраняет результат в формате Parquet.

    Параметры:
    - path (str): Путь к файлам данных.
    - save_path (str): Путь для сохранения объединенных данных.

    Возвращает:
    - None

    Пример использования:
    data_path = 'path/to/data'
    save_path = 'path/to/save/combined_data.parquet'
    concat_all_files_spring(data_path, save_path)
    # Сохраняет объединенные данные в формате Parquet.

    """
    create_dir(save_path.split("\\")[0])
    df = dd.concat(
        [dd.read_parquet(f, engine="pyarrow") for f in glob(path, recursive=True)],
        ignore_index=True,
    )
    df = df.astype(
        {
            "country": str,
            "oblast": str,
            "raion": str,
            "Species": str,
            "Family": str,
            "Bird_id": str,
            "project": str,
            "Sex": str,
            "Age": str,
        }
    )
    # df = df.loc[df['Species'].isin(exclude_list)==False]
    df["Sex"] = df["Sex"].fillna("Unknown")
    df["Age"] = df["Age"].fillna("adult")
    df["year"] = df["timestamp"].dt.year
    df["month"] = df["timestamp"].dt.month
    df = df.loc[(df["month"] >= 2) & (df["month"] <= 6), :]  # (df['country']=='Россия')
    df = df.drop_duplicates(
        subset=[
            "event_id",
            "timestamp",
            "location_long",
            "location_lat",
            "ground_speed",
            "heading",
            "height_above_msl",
            "country",
            "oblast",
            "raion",
            "hunting",
            "Bird_id",
            "Species",
        ],
        ignore_index=True,
    )
    df.compute().to_parquet(save_path)


def choose_first_date_ru(raw, first_date):
    """
    Выбирает данные, учитывая первую дату в России для каждого экземпляра.

    Параметры:
    - raw (pandas DataFrame): Исходные данные.
    - first_date (pandas DataFrame): DataFrame с первыми датами в России для каждого экземпляра.

    Возвращает:
    - merged (pandas DataFrame): DataFrame с выбранными данными, учитывая первую дату в России.

    Пример использования:
    raw_data = pd.DataFrame(...)
    first_dates = pd.DataFrame(...)
    selected_data = choose_first_date_ru(raw_data, first_dates)
    # Возвращает DataFrame с выбранными данными, учитывая первую дату в России.

    """
    merged = raw.merge(
        first_date,
        on=[
            "Species",
            "Bird_id",
            # 'Sex',
            "year",
            "country",
        ],
        how="left",
    )

    # merged = merged[(merged['first_day_in_russia'] < merged['timestamp'])]
    return merged


def calculate_bearing(lat1, lon1, lat2, lon2):
    """
    Calculate the bearing between two points on the earth.

    Parameters:
    lat1 (float): Latitude of the starting point in decimal degrees.
    lon1 (float): Longitude of the starting point in decimal degrees.
    lat2 (float): Latitude of the ending point in decimal degrees.
    lon2 (float): Longitude of the ending point in decimal degrees.

    Returns:
    float: Bearing in degrees from the starting point to the ending point.
    """
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Difference in longitude
    delta_lon = lon2 - lon1

    # Calculate bearing
    x = np.sin(delta_lon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - (np.sin(lat1) * np.cos(lat2) * np.cos(delta_lon))

    # Convert bearing from radians to degrees
    initial_bearing = np.arctan2(x, y)
    initial_bearing = np.degrees(initial_bearing)

    # Normalize the bearing to 0-360 degrees
    compass_bearing = (initial_bearing + 360) % 360

    return compass_bearing


def last_day_of_migration(data):
    """
    Analyze bird migration data to calculate the bearing and identify the last day of
    north or north-east movement for each bird each year. This version takes into account
    that the last day of migration is the last flight in which the bird flew in the
    direction of the north or northeast for less than an hour.

    Parameters:
    data (DataFrame): The bird migration data.

    Returns:
    DataFrame: The original data enhanced with 'bearing' and 'last_date' columns.
    """
    # Exclude data when birds are at stops
    # data = data[data["date"] <= pd.to_datetime(data["year"].astype(str) + "-06-01")]
    data = data[data["stops"] == False]

    # Group data by Bird_id, year, and hour
    grouped_data = data.groupby(["Bird_id", "year", "date", data["timestamp"].dt.hour])

    # Calculate average coordinates for each group
    avg_coords = grouped_data.agg(
        {"location_lat": "mean", "location_long": "mean"}
    ).reset_index()

    # Calculate bearings for each hourly interval
    avg_coords["bearing"] = calculate_bearing(
        avg_coords["location_lat"],
        avg_coords["location_long"],
        avg_coords["location_lat"].shift(-1),
        avg_coords["location_long"].shift(-1),
    )

    # Identify valid north/northeast flights
    valid_flights = avg_coords[
        ((avg_coords["bearing"] >= 0) & (avg_coords["bearing"] <= 60))
        | ((avg_coords["bearing"] >= 340) & (avg_coords["bearing"] <= 360))
    ]

    # Find the last date of valid flights for each bird and year
    last_valid_flights = (
        valid_flights.groupby(["Bird_id", "year"])["date"].max().reset_index()
    )
    last_valid_flights.rename(columns={"date": "last_date"}, inplace=True)

    # Join the last movement dates with the original data
    # data = pd.merge(data, last_valid_flights, on=['Bird_id', 'year'], how='left')

    return last_valid_flights


def first_day_in_ru(df):
    """
    Вычисляет первую дату в России для каждого экземпляра и выбирает данные, учитывая эту дату.

    Пример использования:
    data = pd.DataFrame(...)
    selected_data = first_day_in_ru(data)
    # Возвращает DataFrame с выбранными данными, учитывая первую дату в России для каждого экземпляра.

    """
    df = df.sort_values(by=["Bird_id", "date"])
    df = df[
        # (df["month"] < 6)
        (df["country"] != "None")
        & (df["oblast"] != "Калининградская область")
    ]
    # Create a shifted country column to detect border crossing
    df["prev_country"] = df.groupby(["Bird_id", "date"])["country"].shift(1)

    # Detect a change in country to Russia
    df["to_russia"] = (
        (df["country"] == "Россия")
        & (df["prev_country"] != "Россия")
        & (~df["prev_country"].isna())
    )

    # Определяем начало и конец пересечения границы
    df["cross_start"] = df["to_russia"] & (df["country"] == "Россия")
    df["cross_end"] = df["to_russia"] & ~(df["country"] == "Россия")

    # Группируем по Bird_id, year, Species и вычисляем продолжительность пересечения
    cross_duration = df[df["cross_start"] | df["cross_end"]].copy()
    cross_duration["timestamp_next"] = cross_duration.groupby(
        ["Bird_id", "year", "Species"]
    )["timestamp"].shift(-1)
    cross_duration["duration"] = (
        cross_duration["timestamp_next"] - cross_duration["timestamp"]
    )
    cross_duration["duration_days"] = (
        cross_duration["duration"].dt.total_seconds() / 86400
    )
    cross_duration = cross_duration[cross_duration["cross_start"]]

    # Выбираем те, что дольше 5 дней
    cross_duration = cross_duration[cross_duration["duration_days"] > 5]

    # Group by bird, year, and to_russia, then get the first appearance with crossing
    first_appearances = (
        cross_duration.groupby(["Bird_id", "year"])["date"].min().reset_index()
    )
    first_appearances.rename(columns={"date": "first_day_in_russia"}, inplace=True)
    birds_with_crossing = first_appearances.dropna(subset="first_day_in_russia")[
        ["Bird_id", "year", "first_day_in_russia"]
    ]
    # Group by bird, year, and to_russia, then get the first appearance without crossing
    first_appearances_no_crossing = (
        df[(df["month"] >= 3) & (df['to_russia'] == True)].groupby(["Bird_id", "year"])["date"].min().reset_index()
    )
    first_appearances_no_crossing.rename(
        columns={"date": "first_day_in_russia"}, inplace=True
    )

    appearances = pd.concat(
        [birds_with_crossing, first_appearances_no_crossing]
    ).drop_duplicates(subset=["Bird_id", "year"], keep="first")
    first_day_df = df.merge(appearances, on=["Bird_id", "year"], how="left")
    last_day_df = last_day_of_migration(df)
    merged = first_day_df.merge(last_day_df, on=["Bird_id", "year"], how="left")
    merged["first_day_of_year_in_russia"] = merged["first_day_in_russia"].dt.dayofyear

    return merged[
        (merged["date"] >= merged["first_day_in_russia"])
        & (merged["date"] <= merged["last_date"])
    ]


def time_in_country(first_day, hunt=False):
    """
    Вычисляет общую продолжительность пребывания в стране для каждого экземпляра.

    Параметры:
    - first_day (pandas DataFrame): DataFrame с первыми датами в России для каждого экземпляра.
    - hunt (bool, по умолчанию False): Флаг, указывающий, учитывать ли охоту при вычислении продолжительности пребывания.

    Возвращает:
    - out (pandas DataFrame): DataFrame с общей продолжительностью пребывания в стране для каждого экземпляра.

    Пример использования:
    first_dates = pd.DataFrame(...)
    result = time_in_country(first_dates, hunt=True)
    # Возвращает DataFrame с общей продолжительностью пребывания в стране для каждого экземпляра.

    """
    duration_with_first_day = first_day.copy(deep=True)
    duration_with_first_day = duration_with_first_day.loc[
        (
            duration_with_first_day["first_day_in_russia"]
            <= duration_with_first_day["date"]
        )
        & (duration_with_first_day["date"] <= duration_with_first_day["last_date"]),
        :,
    ]
    if hunt:
        grouping_list = [
            "Species",
            "Bird_id",
            # 'Sex',
            "year",
            "hunting",
        ]

        duration_with_first_day_gr = (
            duration_with_first_day.groupby(grouping_list, as_index=False)[
                "hours_in_raion"
            ]
            .sum()
            .rename(columns={"hours_in_raion": "Total days in country"})
        )
        duration_with_first_day_gr["Total days in country"] = (
            duration_with_first_day_gr["Total days in country"] / 24
        )
        duration_with_first_day_gr.loc[
            duration_with_first_day_gr["Total days in country"] < 1,
            "Total days in country",
        ] = (
            duration_with_first_day_gr.loc[
                duration_with_first_day_gr["Total days in country"] < 1,
                "Total days in country",
            ]
            + 1
        )
        return duration_with_first_day_gr
    else:
        grouping_list = [
            "Species",
            "Bird_id",
            # 'Sex',
            "year",
        ]
        duration_with_first_day["Total days in country"] = (
            duration_with_first_day.loc[:, ["first_day_in_russia", "last_date"]]
            .diff(axis=1)["last_date"]
            .dt.days
        )
        out = duration_with_first_day.groupby(grouping_list, as_index=False)[
            "Total days in country"
        ].first()
        return out


def ratio_hunt_total_day(df):
    """
    Вычисляет отношение количества дней охоты к общему количеству дней в стране.

    Аргументы:
        df (DataFrame): Исходные данные, содержащие переменные.

    Возвращает:
        DataFrame: Данные с отношением дней охоты к общему количеству дней в стране.
    """
    ratio = (
        df.pivot_table(
            index=[
                "Species",
                "Bird_id",
                # 'Sex',
                "year",
            ],
            columns="hunting",
            values="Total days in country",
        )
        .reset_index()
        .rename(columns={True: "hunt_true", False: "hunt_false"})
    )
    ratio["hunt_true"].fillna(0, inplace=True)
    ratio["hunt_day/tot_day"] = ratio["hunt_true"] / ratio.loc[
        :, ["hunt_true", "hunt_false"]
    ].sum(axis=1)
    return ratio


def stop_num_count(df, cols_to_groupby):
    """
    Подсчитывает количество смен состояния остановки (количество остановок) в данных.

    Аргументы:
        df (DataFrame): Исходные данные, содержащие переменные.
        cols_to_groupby (list): Список столбцов, по которым выполняется группировка.

    Возвращает:
        DataFrame: Данные с подсчитанным количеством смен состояния остановки.
    """
    # Вычисляем моменты, когда статус остановки меняется с False на True
    df["start_stop"] = df["stops"] & (~df["stops"].shift(1, fill_value=False))

    # Подсчитываем количество остановок для каждой птицы за каждый год
    stops_count = (
        df.groupby(["Species", "Bird_id", "year", "hunting"])["start_stop"]
        .sum()
        .reset_index()
    )

    # Переименовываем столбец для лучшего понимания
    stops_count.rename(columns={"start_stop": "stops_number"}, inplace=True)

    return stops_count


def calculate_stops_duration(df):
    """
    Вычисляет среднюю продолжительность остановок в данных.

    Аргументы:
        df (DataFrame): Исходные данные, содержащие переменные.

    Возвращает:
        DataFrame: Данные со средней продолжительностью остановок.
    """
    # Определяем изменения в статусе остановок
    df["stop_change"] = df["stops"].diff().ne(0)

    # Определяем начало и конец стоянок
    df["stop_start"] = df["stop_change"] & df["stops"]
    df["stop_end"] = df["stop_change"] & ~df["stops"]

    # Учитываем последнюю запись в данных для каждой птицы
    df["last_record"] = ~df[["Bird_id", "year"]].duplicated(keep="last")

    # Устанавливаем конец стоянки для последней записи, если она означает стоянку
    df.loc[df["stops"] & df["last_record"], "stop_end"] = True

    # Группируем по Bird_id, year, Species и вычисляем продолжительность стоянок
    stops_duration = df[df["stop_start"] | df["stop_end"]].copy()
    stops_duration["timestamp_next"] = stops_duration.groupby(
        ["Bird_id", "year", "Species"]
    )["timestamp"].shift(-1)
    stops_duration["duration"] = (
        stops_duration["timestamp_next"] - stops_duration["timestamp"]
    )
    stops_duration["duration_days"] = (
        stops_duration["duration"].dt.total_seconds() / 86400
    )
    stops_duration = stops_duration[stops_duration["stop_start"]]

    # Разделяем стоянки на во время охоты и вне охоты
    stops_during_hunting = stops_duration[stops_duration["hunting"]]
    stops_outside_hunting = stops_duration[~stops_duration["hunting"]]

    # Выбираем нужные столбцы
    stops_during_hunting = stops_during_hunting[
        ["Bird_id", "year", "Species", "timestamp", "timestamp_next", "duration_days"]
    ]
    stops_outside_hunting = stops_outside_hunting[
        ["Bird_id", "year", "Species", "timestamp", "timestamp_next", "duration_days"]
    ]

    stops_during_hunting["hunting"] = True
    stops_outside_hunting["hunting"] = False
    stops_outside_hunting["duration_days"] = np.ceil(
        stops_outside_hunting["duration_days"]
    )
    out = pd.concat([stops_during_hunting, stops_outside_hunting])
    return out.rename(columns={"duration_days": "Mean stopover duration, days"}).loc[
        :, ["Species", "Bird_id", "year", "Mean stopover duration, days", "hunting"]
    ]


def speed_dist_grouping(df):
    """
    Группирует данные по виду птицы, идентификатору птицы, полу, году и охоте,
    и вычисляет суммарное время нахождения в районе, суммарное расстояние
    и среднюю скорость перемещения.

    Аргументы:
        df (DataFrame): Исходные данные, содержащие переменные.

    Возвращает:
        DataFrame: Группированные данные с суммарным временем, суммарным расстоянием и средней скоростью.
    """

    df = (
        df[df["stops"] == False]
        .groupby(
            [
                "Species",
                "Bird_id",
                # 'Sex',
                "year",
                "hunting",
            ],
            as_index=False,
        )
        .agg({"hours_in_raion": "sum", "dist, km": "sum"})
    )
    df["V, km/h"] = df["dist, km"] / df["hours_in_raion"]
    return df


def speed_stops_grouping(df):
    """
    Группирует данные по виду птицы, идентификатору птицы, полу, году, состоянию остановки и охоте,
    и вычисляет суммарное время нахождения в районе, суммарное расстояние
    и среднюю скорость перемещения.

    Аргументы:
        df (DataFrame): Исходные данные, содержащие переменные.

    Возвращает:
        DataFrame: Группированные данные с суммарным временем, суммарным расстоянием и средней скоростью.
    """
    df = df.groupby(
        [
            "Species",
            "Bird_id",
            # 'Sex',
            "year",
            "stops",
            "hunting",
        ],
        as_index=False,
    ).agg({"hours_in_raion": "sum", "dist, km": "sum"})
    df["V, km/h"] = df["dist, km"] / df["hours_in_raion"]
    return df


# Визуализация


def find_dead_birds(df):
    """
    Фильтрует DataFrame с данными о птицах, исключая птиц, которые считаются мертвыми на основе определенных критериев.

    Параметры:
    df (pandas.DataFrame): Входной DataFrame, содержащий данные о птицах с колонками 
    'Bird_id', 'country', 'year', 'date', 'dist, km' и 'Species'.

    Возвращает:
    pandas.DataFrame: Отфильтрованный DataFrame, содержащий только записи о птицах,
    которые не считаются мертвыми на основе заданных критериев.

    Описание:
    Эта функция принимает на вход DataFrame 'df', который представляет собой данные о перемещении птиц.
    Она выполняет следующие шаги для исключения птиц, считающихся мертвыми:

    1. Агрегирует данные, группируя их по 'Bird_id', 'country', 'year' и 'date', вычисляя общее расстояние,
    пройденное каждой птицей в каждый день.

    2. Фильтрует агрегированные данные, включая только записи,
    где общее расстояние ('dist, km') меньше или равно 0.05 километра, и где страна - 'Россия'.

    3. Дополнительно агрегирует отфильтрованные данные по 'Bird_id', 'country' и 'year',
    подсчитывая количество дней, в которые каждая птица удовлетворяет критериям из шага 2.

    4. Создает список 'exclude_list' значений 'Bird_id' для птиц, которые соответствуют критерию,
    имея более одного дня с расстоянием менее или равным 0.05 километра в России.

    5. Создает директорию для хранения таблиц, связанных с исключенными птицами,
    специфичную для уникальных видов во входном DataFrame.

    6. Возвращает отфильтрованный DataFrame, исключая птиц, чей 'Bird_id' находится в 'exclude_list'.

    Пример:
    Для использования этой функции можно передать DataFrame, содержащий данные о перемещении птиц, следующим образом:

    filtered_df = find_dead_birds(input_df)

    В 'filtered_df' будут содержаться только записи о птицах,
    которые не удовлетворяют критериям для идентификации как мертвые.
    """
    # dist_by_day = df.groupby(
    #     ["Bird_id", "country", "year", "date"], as_index=False
    # ).agg({"dist, km": "sum"})
    # count_dead_days = (
    #     dist_by_day[
    #         (dist_by_day["dist, km"] <= 0.05) & (dist_by_day["country"] == "Россия")
    #     ]
    #     .groupby(["Bird_id", "country", "year"], as_index=False)["date"]
    #     .count()
    # )
    # exclude_list = count_dead_days[count_dead_days["date"] > 1]["Bird_id"].unique()
    # os.makedirs(
    #     f"tables_for_fig_without_Hungary\\excluded_birds\\excluded{df['Species'].unique()}",
    #     exist_ok=True,
    # )
    # return df[~df["Bird_id"].isin(exclude_list)]

    dist_by_day = df.groupby(
        ["Bird_id", "country", "year", "date"], as_index=False
    ).agg({"dist, km": "sum"})

    count_dead_days = (
        dist_by_day[
            # (dist_by_day["dist, km"] <= 0.05) & (dist_by_day["country"] == "Россия")
            (dist_by_day["dist, km"] <= 0.05)
        ]
        .groupby(["Bird_id", "country", "year"], as_index=False)["date"]
        .count()
    )

    exclude_list = count_dead_days[count_dead_days["date"] > 1]["Bird_id"].unique()
    # save_path = "data\\dead_birds\\"
    # os.makedirs(save_path, exist_ok=True)

    # np.savetxt(
    #     f"{save_path}\\excluded_{df['Species'].unique()[0]}.txt",
    #     exclude_list,
    #     delimiter="\t",
    #     fmt="%s",
    # )

    return df[~df["Bird_id"].isin(exclude_list)]


def sample_size_measure(parquet_path):
    """
    Измеряет размер выборки для каждого вида на основе данных в формате Parquet.

    Параметры:
    - parquet_path (list): Список путей к файлам данных в формате Parquet для каждого вида.

    Возвращает:
    - sample_size (pandas DataFrame): DataFrame с измерениями размеров выборки для каждого вида.

    Пример использования:
    parquet_files = ['path/to/parquet/file1.parquet',
        'path/to/parquet/file2.parquet', 'path/to/parquet/file3.parquet']
    sample_sizes = sample_size_measure(parquet_files)
    # Возвращает DataFrame с измерениями размеров выборки для каждого вида.

    """

    def each_sp_sample_size_measure(path, species):
        df = pd.read_parquet(path)
        df = find_dead_birds(df)
        return (
            df.groupby(["year"])["Bird_id"]
            .nunique()
            .to_frame()
            .rename(columns={"Bird_id": species})
        )

    sample_size = (
        each_sp_sample_size_measure(parquet_path[0], "Белолобые гуси")
        .join(each_sp_sample_size_measure(parquet_path[1], "Казарки"))
        .join(each_sp_sample_size_measure(parquet_path[2], "Гуменники"))
    )
    return sample_size.astype("Int64", errors="ignore")


def pairs_generator(data, x, hue):
    """
    Генерирует пары значений для заданных столбцов в DataFrame.

    Параметры:
    - data (pandas DataFrame): Исходный DataFrame.
    - x (str): Название столбца, для которого генерируются значения.
    - hue (str): Название столбца, по которому группируются значения.

    Возвращает:
    - pairs (list): Список пар значений для каждого значения столбца x и hue.

    Пример использования:
    data = pd.DataFrame(
        {'x': [1, 2, 3, 4, 5], 'hue': ['A', 'A', 'B', 'B', 'C']})
    pairs = pairs_generator(data, 'x', 'hue')
    # Возвращает список пар значений, сгруппированных по столбцам 'x' и 'hue'.

    """
    x_point_list = data[x].sort_values().unique()
    x_pairs = list(product(x_point_list, data[hue].unique()))
    return [x_pairs[i: i + 2] for i in range(0, len(x_pairs), 2)]


def stat_data_extraction(test_res, sp, feature):
    """
    Извлекает статистические данные из результатов тестирования и формирует DataFrame.

    Параметры:
    - test_res (list): Список объектов с результатами тестирования.
    - sp (str): Вид птицы.
    - feature (str):
    feature:

    Возвращает:

    df_out (pandas DataFrame): DataFrame с извлеченными статистическими данными.
    Пример использования:
    test_results = [...] # Список объектов с результатами тестирования
    species = "Sparrow" # Вид птицы
    feature = "Weight" # Характеристика
    extracted_data = stat_data_extraction(test_results, species, feature)

    Возвращает DataFrame с извлеченными статистическими данными, включая вид птицы и характеристику.
    """
    g1, g2, index = [], [], []
    test_name, stat_val, p_val = [], [], []
    for res in test_res:
        gr1_name = (
            str(res.data.group1)
            .replace(",", "")
            .replace("(", "")
            .replace(")", "")
            .replace(" ", "_")
        )
        g2_name = (
            str(res.data.group2)
            .replace(",", "")
            .replace("(", "")
            .replace(")", "")
            .replace(" ", "_")
        )
        g1.append(gr1_name)
        g2.append(g2_name)
        index.append(f"{gr1_name} vs {g2_name}")
        test_name.append(res.data.test_short_name)
        stat_val.append(res.data.stat_value)
        p_val.append(res.data.pvalue)
    df_out = pd.DataFrame(
        index=index,
        data=list(zip(test_name, stat_val, p_val)),
        columns=["Test", "U-stat", "p-value"],
    )
    df_out["Species"] = sp
    df_out["Feature"] = feature
    return df_out


def boxlot_stats(
    data, x, feature, hue, figure_path, figname, stat_data, test="Mann-Whitney"
):
    """
    Создает boxplot с аннотациями статистических тестов и сохраняет ее в файл.

    Параметры:
    - data (pandas DataFrame): Исходный DataFrame с данными.
    - x (str): Название столбца, отображаемого по оси x.
    - feature (str): Название столбца, отображаемого по оси y (характеристика).
    - hue (str): Название столбца, по которому группируются данные.
    - figure_path (str): Путь к директории, в которую будет сохранена ящиковая диаграмма.
    - figname (str): Название для сохраняемой ящиковой диаграммы.

    Возвращает:
    - df_out (pandas DataFrame): DataFrame с извлеченными статистическими данными.

    Пример использования:
    data = pd.DataFrame(...)
    figure_directory = "path/to/figures"
    statistical_results = boxlot_stats(
        data, 'x', 'feature', 'hue', figure_directory, 'my_figure')
    # Создает boxplot, сохраняет ее в файл 'my_figure.png' и возвращает статистические результаты.

    """
    sns.set(rc={"figure.figsize": (10, 6)})
    sns.set_theme(font_scale=1.5, style="ticks", context="notebook", palette="Set3")
    sns.set_style("whitegrid")
    hue_order = [True, False]
    plot = sns.boxplot(
        data=data,
        x=x,
        y=feature,
        hue=hue,
        # hue_order=hue_order,
        showfliers=False,
    )
    plot.yaxis.set_major_locator(ticker.MaxNLocator(nbins=8))
    pairs = pairs_generator(data, x, hue)

    annotator = Annotator(plot, pairs, data=data, x=x, y=feature, hue=hue)
    annotator.configure(test=test, loc="inside")
    _, test_results = annotator.apply_test().annotate()

    plot.set_title(figname.split("_")[0])
    plt.legend(loc=2, bbox_to_anchor=(1.03, 1), borderaxespad=0, title=hue.capitalize())
    create_dir(figure_path)
    plt.savefig(f"{figure_path}\\{figname}.png", bbox_inches="tight")
    plt.show()
    plt.close("all")
    table_save_path = figure_path.split("\\", 1)[1]
    create_dir(f"tables_for_fig\\{table_save_path}")
    data.to_excel(f"tables_for_fig\\{table_save_path}\\{figname}.xlsx")
    return pd.concat(
        [
            stat_data,
            stat_data_extraction(test_results, data["Species"].unique()[0], figname),
        ]
    )


def boxlot_stats_year_comparison(
    data, x, feature, figure_path, figname, stat_data, test="Mann-Whitney"
):
    """
    Генерирует boxlot с статистическими аннотациями и выполняет сравнение по годам.

    Аргументы:
        data (DataFrame): Исходные данные, содержащие переменные.
        x (str): Название столбца в 'data', который будет отображаться на оси X.
        feature (str): Название столбца в 'data', который будет отображаться на оси Y.
        figure_path (str): Путь к директории, где будет сохранена графика.
        figname (str): Название файла с графикой.

    Возвращает:
        DataFrame: Статистические данные, извлеченные из аннотаций.
    """
    sns.set(rc={"figure.figsize": (7, 10)})
    sns.set_theme(font_scale=1.5, style="ticks", context="notebook", palette="Set3")
    sns.set_style("whitegrid")
    hue_order = [True, False]
    plot = sns.boxplot(data=data, x=x, y=feature, showfliers=False)
    plot.yaxis.set_major_locator(ticker.MaxNLocator(nbins=20))

    def window(iterable, size):
        iterators = tee(iterable, size)
        iterators = [islice(iterator, i, None) for i, iterator in enumerate(iterators)]
        yield from zip(*iterators)

    pairs = list(window(data[x].sort_values().unique(), 2))

    annotator = Annotator(plot, pairs, data=data, x=x, y=feature)
    annotator.configure(
        test=test,
        loc="inside",
        # comparisons_correction="Bonferroni"
        # comparisons_correction='Holm-Bonferroni',
    )
    _, test_results = annotator.apply_test().annotate(
        line_offset_to_group=0.2, line_offset=0.2
    )

    plot.set_title(figname.split("_")[0])
    create_dir(figure_path)
    plt.savefig(f"{figure_path}\\{figname}.png", bbox_inches="tight")
    plt.show()
    plt.close("all")
    table_save_path = figure_path.split("\\", 1)[1]
    create_dir(f"tables_for_fig\\{table_save_path}")
    data.to_excel(f"tables_for_fig\\{table_save_path}\\{figname}.xlsx")
    return pd.concat(
        [
            stat_data,
            stat_data_extraction(test_results, data["Species"].unique()[0], figname),
        ]
    )


def read_dead_birds(file_pattern, delimiter="\t", dtype=None):
    """
    Читает текстовые файлы, соответствующие заданному шаблону, и объединяет их данные в один массив.

    Параметры:
    file_pattern (str): Шаблон для поиска файлов с помощью glob.glob.
    delimiter (str): Разделитель данных в текстовых файлах.
    dtype (dtype, optional): Тип данных, в который нужно преобразовать считанные значения. По умолчанию None.

    Возвращает:
    numpy.ndarray: Объединенный массив данных из всех найденных файлов.

    Пример использования:
    data = read_text_files_and_concatenate('*.txt', delimiter='\t', dtype=float)
    """
    file_list = glob(file_pattern)  # Находим файлы, соответствующие шаблону

    arrays = [
        np.loadtxt(file, delimiter=delimiter, dtype=dtype) for file in file_list
    ]  # Читаем данные из файлов

    concatenated_array = np.concatenate(arrays)  # Объединяем данные в один массив

    return concatenated_array


def find_similar_bird_id(data):
    # Convert list to numpy array
    data_array = np.array(data)

    # Regular expression to extract the base part of each string
    base_part = np.vectorize(lambda x: re.match(r'(.+?)(-\d+)?$', x).group(1))(data_array)

    # Finding pairs: compare each element's base part with every other element's base part
    # Using broadcasting
    match_matrix = base_part[:, None] == base_part

    # Exclude exact matches (diagonal)
    np.fill_diagonal(match_matrix, False)

    # Find indices where matches occur
    match_indices = np.where(match_matrix)

    # Extract pairs based on indices
    similar_pairs = [(data_array[i], data_array[j]) for i, j in zip(*match_indices) if i < j]

    # Create a string to save in file
    # text_to_save = "\n".join([f"{pair[0]} {pair[1]}" for pair in similar_pairs])

    # Define file path
    # file_path = 'similar_birds.txt'

    # Write to file
    # with open(file_path, 'w') as file:
    #     file.write(text_to_save)

    return pd.DataFrame(similar_pairs).loc[:, 1]


def find_days_with_no_signal(data):
    """
    Функция для определения дней, когда отсутствовал сигнал для каждой птицы в каждом году.

    :param data: DataFrame с данными о птицах и сигналах, содержащий столбцы 'bird_id', 'year' и 'date'.
    :return: DataFrame с идентификаторами птиц, годами и датами, когда отсутствовал сигнал.
    """
    data['date'] = pd.to_datetime(data['date']).dt.date
    # Получение минимальной даты и установка максимальной даты (1 июля) для каждой птицы в каждом году
    # data['first_day_in_russia'] = pd.to_datetime(data['first_day_in_russia']).dt.date
    # сохраним птиц, для которых нет даты окончания миграции, т.е. имелись перерывы в работе датчика
    # save_path = f"tables_for_fig_without_Hungary\\excluded_birds\\"
    # np.savetxt(f"{save_path}\\excluded_{data['Species'].unique()[0]}_no_last_day.txt", 
    #            #data.loc[data['last_date'].isna(), 'Bird_id'].unique(),
    #            data['Bird_id'].unique(),  
    #            delimiter = '\t',
    #            fmt='%s')

    # data = data[~data['last_date'].isna()]
    date_ranges = data.groupby(['Bird_id', 'year'])['date'].min().reset_index()
    # Обработка пропущенных значений в last_date
    date_ranges['max_date'] = pd.to_datetime(date_ranges['year'].astype(str) + '-05-10').dt.date
    # print(date_ranges.columns)
    # Генерация всех возможных дат для каждой комбинации Bird_id и year
    all_dates = pd.concat([
        pd.DataFrame({'Bird_id': bird_id, 'year': year, 'date': pd.date_range(start=min_date, end=max_date, freq='D').date})
        for bird_id, year, min_date, max_date in date_ranges.to_numpy()
    ])
    # Проверка наличия каждой даты в исходных данных
    merged = pd.merge(all_dates, data, on=['Bird_id', 'year', 'date'], how='left', indicator=True)
    missing_dates = merged[merged['_merge'] == 'left_only']
    missing_dates = missing_dates.groupby(['Bird_id', 'year'], as_index = False)['date'].count()
    # missing_dates.to_excel(f"tables_for_fig_without_Hungary\\excluded_birds\\excl_by_missing_signal_{data['Species'].unique()[0]}.xlsx")
    missing_dates = missing_dates[missing_dates['date']>10]

    return missing_dates[['Bird_id', 'year']].drop_duplicates()


def exclude_short_visit_birds(df_to_filter, df_filter, columns):
    """
    Исключает из df_to_filter строки, которые совпадают с комбинациями в df_filter по указанным столбцам.

    :param df_to_filter: DataFrame, из которого нужно исключить строки.
    :param df_filter: DataFrame, содержащий комбинации для исключения.
    :param columns: Список столбцов, по которым происходит сравнение.
    :return: Отфильтрованный DataFrame.
    """
    # Объединение df_to_filter с df_filter для определения совпадающих строк
    merged_df = df_to_filter.merge(df_filter, on=columns, how='left', indicator=True)

    # Отфильтровываем те строки, которые не совпадают с df_filter
    filtered_df = merged_df[merged_df['_merge'] == 'left_only'].drop(columns=['_merge'])

    return filtered_df
