from PIL import Image
from operator import itemgetter
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def deg2num(latitude, longitude, zoom, do_round=True):
    """Convert from latitude and longitude to tile numbers.
    If do_round is True, return integers. Otherwise, return floating point
    values.
    Source: http://wiki.openstreetmap.org/wiki/Slippy_map_tilenames#Python
    """
    lat_rad = np.radians(latitude)
    n = 2.0 ** zoom
    if do_round:
        f = np.floor
    else:
        f = lambda x: x
    xtile = f((longitude + 180.) / 360. * n)
    ytile = f((1.0 - np.log(np.tan(lat_rad) + (1 / np.cos(lat_rad))) / np.pi) / 2. * n)
    if do_round:
        if isinstance(xtile, np.ndarray):
            xtile = xtile.astype(np.int32)
        else:
            xtile = int(xtile)
        if isinstance(ytile, np.ndarray):
            ytile = ytile.astype(np.int32)
        else:
            ytile = int(ytile)
    return (xtile, ytile)

def get_tile_coords(lat, lon, z):
    """Convert geographical coordinates to tile coordinates (integers),
    at a given zoom level."""
    return deg2num(lat, lon, z, do_round=False)

def to_pixels(lat, lon, z, box_tile):
    """Convert from geographical coordinates to pixels in the image."""
    TILE_SIZE = 256 # constant
    x, y = get_tile_coords(lat, lon, z)
    xmin = min(box_tile[0], box_tile[2])
    ymin = min(box_tile[1], box_tile[3])
    px = (x - xmin) * TILE_SIZE
    py = (y - ymin) * TILE_SIZE
    return px, py


taxi_id_file = "./cabspottingdata/_cabs.txt" #全タクシーのIDが記録されたファイル
id_list_of_taxis = []
with open(taxi_id_file, "r") as f:
    for line in f:
        line = line.rstrip()
        taxi_id = line.split('"')[1]
        id_list_of_taxis.append(taxi_id)

pos0 = (37.57708, -122.51975) #左下
pos1 = (37.83071, -122.35057) #右上
num_to_divide_x = 100
interval = abs(pos0[1] - pos1[1]) / num_to_divide_x
num_to_divide_y = int(abs(pos0[0] - pos1[0]) // interval)
pos1 = (pos0[0] + num_to_divide_y * interval, pos1[1])
#x_interval = abs(pos0[1] - pos1[1]) / n
#y_interval = abs(pos0[0] - pos1[0]) / n
leftmost_lng = pos0[1]
upmost_lat = pos1[0]

passenger_distribution = [[0 for i in range(num_to_divide_x)] for j in range(num_to_divide_y)]

for taxi_id in id_list_of_taxis:
    filename = "./cabspottingdata/" + "new_" + taxi_id + ".txt"
    data_list = [] #unixtimeでソートするために、一時的にデータを読み込む
    with open(filename, "r") as f:
        for line in f:
            line = line.rstrip()
            lat, lng, ride_state, unixtime = line.split()
            lat, lng = float(lat), float(lng)
            ride_state, unixtime = int(ride_state), int(unixtime)
            data_list.append([lat, lng, ride_state, unixtime])
    sorted_data_list = sorted(data_list, key=itemgetter(3))
    is_first_line = True
    for lat, lng, ride_state, unixtime in sorted_data_list:
        if is_first_line:
            prev_lat, prev_lng, prev_ride_state, prev_unixtime = lat, lng, ride_state, unixtime
            is_first_line = False
            continue
        if ride_state == 0 and prev_ride_state == 1:
            x_coordinate = int((lng - leftmost_lng) // interval)
            y_coordinate = int((upmost_lat - lat) // interval)
            #print(x_coordinate,y_coordinate)
            if 0 <= x_coordinate < num_to_divide_x and 0 <= y_coordinate < num_to_divide_y:
                passenger_distribution[y_coordinate][x_coordinate] += 1
        prev_lat, prev_lng, prev_ride_state, prev_unixtime = lat, lng, ride_state, unixtime



print(passenger_distribution)


fig = plt.figure(dpi=300)
ax1 = fig.add_subplot(111)
#ax1.imshow(img)
sns.heatmap(passenger_distribution)
#sns.heatmap(passenger_distribution, cmap='coolwarm', square=True, robust=True, ax=ax1, center=1000)
#plt.show()
#plt.savefig("asdf.png")
plt.axis('off')
plt.savefig("heatmap.png")
fig.clf()
plt.clf()
plt.close()


