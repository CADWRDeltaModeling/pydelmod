# Functions to help with DSM2 and GIS related tasks
# %%
import math
import pandas as pd
import geopandas as gpd
import shapely
from shapely.geometry import Point, MultiLineString
from shapely.ops import nearest_points


def find_closest_line_and_distance(point: Point, gdf):
    min_distance = float("inf")
    closest_multiline = None

    for idx, row in gdf.iterrows():
        # For each MultiLineString, find the closest point to the specified point
        closest_point = nearest_points(row["geometry"], point)[0]
        # Calculate the distance from the point to this closest point
        distance = point.distance(closest_point)
        # Update minimum distance and closest MultiLineString if this is the closest so far
        if distance < min_distance:
            min_distance = distance
            closest_multiline = row
    return closest_multiline, min_distance


def get_distance_from_start(point: Point, closest_line: gpd.GeoDataFrame):
    point_on_line = nearest_points(closest_line["geometry"], point)
    distance_from_start = closest_line["geometry"].project(point_on_line[0])
    return distance_from_start


def read_stations(file_path):
    stations = pd.read_csv(file_path)
    stations = gpd.GeoDataFrame(
        stations, geometry=[Point(xy) for xy in zip(stations.lon, stations.lat)]
    )
    # Set the original CRS to WGS84 (latitude and longitude)
    stations.crs = "EPSG:4326"
    # Convert the geometries to EPSG:26910
    stations = stations.to_crs(epsg=26910)
    return stations


def get_id_and_distance_from_start(point, gdf):
    closest_line, dist_from_line = find_closest_line_and_distance(point, gdf)
    dist = get_distance_from_start(point, closest_line)
    if math.isclose(closest_line["geometry"].length, dist, abs_tol=1):
        dist = "LENGTH"
    else:
        dist = int(dist)
    return closest_line.id, dist, dist_from_line


def create_stations_output_file(
    stations_file, centerlines_file, output_file, distance_tolerance=100
):
    """
    Create DSM2 channels output compatible file for given stations info (station_id, lat lon)
    and centerlines geojson file (DSM2 channels centerlines) and writing out output_file

    The distance_tolerance is the maximum distance from a line that a station can be to be considered on that line

    The output file can be used to then create the channels file for DSM2 for these stations.
    Parameters
    ----------
    stations_file : str
        Path to the stations file
    centerlines_file : str
        Path to the centerlines file
    output_file : str
        Path to the output file
    distance_tolerance : int
        Maximum distance from a line that a station can be to be considered on that line
        default is 100 (feet, but depends if geojson file units are in feet or meters)
    """
    centerlines = gpd.read_file(centerlines_file)
    stations = read_stations(stations_file)
    station_dist_tuple = []
    for _, station in stations.iterrows():
        id, dist, dist_from_line = get_id_and_distance_from_start(
            station["geometry"], centerlines
        )
        if dist_from_line > distance_tolerance:
            print(
                f"Station {station['station_id']} is not close enough to a line. Distance: {dist_from_line}, Closest line: {id}"
            )
        else:
            print(f"Station {station['station_id']} is on line {id} at distance {dist}")
            station_dist_tuple.append((station["station_id"], id, dist))
    dfstation_dist = pd.DataFrame(
        station_dist_tuple, columns=["NAME", "CHAN_NO", "DISTANCE"]
    )
    print("Writing to hydro compatible format: ", output_file)
    dfstation_dist.to_csv(output_file, index=False, sep=" ")
