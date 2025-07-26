# coding: utf-8
import sys
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#import shapefile
import smopy
from PIL import Image
import datetime as dt
import time
import json
import os

# functions
user_id = 0
# for Map for smopy
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

# main

# smopy setting
pos0 = (37.45708, -122.5197584); pos1 = (37.95071, -121.97057)

#pos0 = (37.56708, -122.4597584); pos1 = (37.86071, -122.14057) #get map
#san francisco
#z = 9
#z = 10
z = 11
#z = 12

# get map 
if not os.path.exists("SAN-basemap-z%d.png" % (z)):
  get_mapobj = True

  mapobj = smopy.Map(pos0, pos1, z=z, margin=0.0,)
  print("box:", mapobj.box)
  print("z:", mapobj.z)
  print("box_tile:", mapobj.box_tile)
  print("xmin:", mapobj.xmin)
  print("ymin:", mapobj.ymin)

  outfile = open('basemap-z%d.json' % (z), "w")
  json.dump({'box0': mapobj.box[0], 'box1': mapobj.box[1], 'box2': mapobj.box[2], 'box3': mapobj.box[3], 'z': z, 'box_tile0': mapobj.box_tile[0], 'box_tile1': mapobj.box_tile[1], 'box_tile2': mapobj.box_tile[2], 'box_tile3': mapobj.box_tile[3], 'xmin': mapobj.xmin, 'ymin': mapobj.ymin}, outfile)
  outfile.close()

  x0, y0 = mapobj.to_pixels(pos0[0], pos0[1])
  x1, y1 = mapobj.to_pixels(pos1[0], pos1[1])

  img = mapobj.fetch()

  # save the fetched map
  img.save('SAN-basemap-z%d.png' %(z))

else:
  get_mapobj = False

  # load image
  img = Image.open("SAN-basemap-z%d.png" % (z), "r")

  # load json
  infile = open("basemap-z%d.json" %(z), "r")
  json_dic = json.load(infile)
  infile.close()

  box = ( float(json_dic['box0']), float(json_dic['box1']), float(json_dic['box2']), float(json_dic['box3']) )
  z = int(json_dic['z'])
  box_tile = ( float(json_dic['box_tile0']), float(json_dic['box_tile1']), float(json_dic['box_tile2']), float(json_dic['box_tile3']) )
  xmin = int(json_dic['xmin'])
  ymin = int(json_dic['ymin'])

  #
  x0, y0 = to_pixels(pos0[0], pos0[1], z, box_tile)
  x1, y1 = to_pixels(pos1[0], pos1[1], z, box_tile)

filelist = "./cabspottingdata_returner/_cabs.txt"
infilename_list = []
with open(filelist,"r") as f:
  # for line in f:
  #   line = line.rstrip().split('"')
  #   filepath = "./cabspottingdata/" + "new_" + line[1] + ".txt"
  #   infilename_list.append(filepath)
  for line in f:
    file_name = line.split('\n')
    infilename_list.append(file_name[0])

if not os.path.exists("./taxi"):
  os.makedirs("taxi")

all_data_list = []

#infilename_list = ["./cabspottingdata/new_abboip.txt"]
for infilename in infilename_list:
  print(infilename)
  infile = open(infilename, "r")
  user_id += 1
  for line in infile:
    data_tmp_list = line.split(" ")
    unixtime = data_tmp_list[3].replace('\r\n','')
    longitude = float(data_tmp_list[1])
    latitude  = float(data_tmp_list[0])
    ride_state = int(data_tmp_list[2])
    data_list = [ int(unixtime), user_id, longitude, latitude, ride_state ]
    all_data_list.append( data_list )
  infile.close()
all_data_sorted_list = sorted(all_data_list)

x_to_pixcel = []
y_to_pixcel = []
label_list = []
line_counter = 0
png_counter = 0

PST = dt.timezone(dt.timedelta(hours=-8), "PST")

colors = ['blanchedalmond', 'hotpink', 'mediumturquoise', 'darkslategrey', 'orchid', 'saddlebrown', 'ivory', 'darkslateblue', 'palevioletred', 'maroon', 'wheat', 'darkslategray', 'darkorchid', 'blue', 'lawngreen', 'lightgoldenrodyellow', 'floralwhite', 'lavender', 'darkturquoise', 'lightgreen', 'darkgreen', 'mediumspringgreen', 'violet', 'y', 'c', 'deepskyblue', 'burlywood', 'olivedrab', 'darkblue', 'lightcoral', 'pink', 'darkviolet', 'deeppink', 'plum', 'crimson', 'lightyellow', 'thistle', 'mistyrose', 'b', 'dimgray', 'gold', 'skyblue', 'turquoise', 'chartreuse', 'mediumpurple', 'darkgoldenrod', 'royalblue', 'grey', 'snow', 'darkseagreen', 'coral', 'darkcyan', 'salmon', 'slategray', 'darkgrey', 'aqua', 'lavenderblush', 'tan', 'moccasin', 'darkgray', 'bisque', 'powderblue', 'navajowhite', 'mediumorchid', 'sienna', 'seashell', 'white', 'fuchsia', 'cornflowerblue', 'lightblue', 'lightgray', 'm', 'rosybrown', 'seagreen', 'mediumslateblue', 'mediumvioletred', 'darkorange', 'oldlace', 'limegreen', 'slateblue', 'mediumaquamarine', 'darkmagenta', 'darkred', 'red', 'w', 'lightsalmon', 'cyan', 'peru', 'lightslategray', 'gray', 'magenta', 'cornsilk', 'lightgrey', 'lightcyan', 'forestgreen', 'darkolivegreen', 'black', 'mediumseagreen', 'lightpink', 'linen', 'firebrick', 'greenyellow', 'beige', 'purple', 'slategrey', 'k', 'lime', 'indigo', 'palegreen', 'r', 'dimgrey', 'orangered', 'lightskyblue', 'sandybrown', 'olive', 'tomato', 'chocolate', 'ghostwhite', 'gainsboro', 'indianred', 'green', 'darkkhaki', 'blueviolet', 'paleturquoise', 'brown', 'silver', 'orange', 'papayawhip', 'yellowgreen', 'midnightblue', 'lightseagreen', 'azure', 'mintcream', 'teal', 'lightsteelblue', 'navy', 'yellow', 'palegoldenrod', 'g', 'aliceblue', 'mediumblue', 'whitesmoke', 'goldenrod', 'lightslategrey', 'darksalmon', 'cadetblue', 'dodgerblue', 'khaki', 'peachpuff', 'steelblue', 'lemonchiffon', 'aquamarine', 'antiquewhite', 'springgreen', 'honeydew']

print("input file loading ...")
for i in range(len(all_data_sorted_list)):

  #
  unixtime_now = all_data_sorted_list[i][0]
  if line_counter == 0:
    unixtime_old = all_data_sorted_list[i][0]

  minute_interval = 1
  if unixtime_now >= unixtime_old+minute_interval*60: # every 1 min.

    # for drawing figures
    fig = plt.figure(dpi=300)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(x0, x1), ylim=(y0, y1))

    ax.imshow(img)
    timestamp_str = str(dt.datetime.fromtimestamp(unixtime_old, PST))[:-6] + " " + str(PST)
    timestamp_str2 = str(dt.datetime.fromtimestamp(unixtime_old+minute_interval*60, PST))[:-6] + " " + str(PST)
    fig.text(0.25, 0.8, "%s - %s" %(timestamp_str, timestamp_str2))
    plt.axis('off')


    for j in range(len(x_to_pixcel)):
      plt.plot(x_to_pixcel[j], y_to_pixcel[j], marker='.', markersize=3, linestyle="", color=colors[label_list[j]%len(colors)])
    plt.savefig("taxi/result-%05d.png" %(png_counter), dpi=300)
    fig.clf()
    plt.clf()
    plt.close()
    png_counter += 1

    unixtime_old = unixtime_now
    x_to_pixcel = []
    y_to_pixcel = []
    label_list = []


  # convert for overlay 
  #if all_data_sorted_list[i][4] == 1:
  if get_mapobj == False:
    x, y = to_pixels(float(all_data_sorted_list[i][3]), float(all_data_sorted_list[i][2]), z, box_tile)
    x_to_pixcel.append(x); y_to_pixcel.append(y)
    label_list.append(all_data_sorted_list[i][1])
  else:
    x, y = mapobj.to_pixels(float(all_data_sorted_list[i][3]), float(all_data_sorted_list[i][2]))
    x_to_pixcel.append(x); y_to_pixcel.append(y)
    label_list.append(all_data_sorted_list[i][1])

  line_counter += 1
print("input file loaded")
print(line_counter)
