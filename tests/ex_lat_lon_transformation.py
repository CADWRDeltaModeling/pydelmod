# %%
import geopandas as gpd
from shapely.geometry import Point

# Sample data
data = {"station_id": ["aamc1"], "lat": [37.771614], "lon": [-122.299398]}

# Create GeoDataFrame
gdf = gpd.GeoDataFrame(
    data,
    geometry=[Point(xy) for xy in zip(data["lon"], data["lat"])],
    crs="EPSG:4326",  # WGS84 CRS
)
gdf.set_crs(epsg=4326, inplace=True)
# Print initial GeoDataFrame
print("Before transformation:")
print(gdf)

# Convert to UTM Zone 10N
gdf_utm = gdf.to_crs(epsg=26910)
gdf.set_crs(epsg=4326, inplace=True)
gdf_utm = gdf.to_crs(epsg=26910)

# Print transformed GeoDataFrame
print("After transformation:")
print(gdf_utm)

# %%
from pyproj import CRS

print(CRS.from_epsg(4326))
print(CRS.from_epsg(26910))
# %%
gdf.set_crs(epsg=4326, inplace=True)
gdf_utm = gdf.to_crs(epsg=26910)
# %%
