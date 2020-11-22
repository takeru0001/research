#!/usr/bin/env python3
# coding: utf-8

# import modules
import xml.etree.ElementTree as ET
import sys
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.animation import FuncAnimation
import math
from math import sin, cos, acos, radians
#from numba.decorators import jit, njit
from PIL import Image, ImageOps
import smopy
import json
import gc
import random
import datetime

from car_tmp import Car
from lane import Lane
from road_segment import RoadSegment
from ride_prob import get_ride_prob_and_reward
import output

earth_rad = 6378.137

# simulation settings
#infilename = "sfc_main.net.xml"
infilename = "SanFrancisco2.net.xml"
#infilename = "sfc_small.net.xml"
#infilename = "sfc.net.xml"


png_infilename = "sanfrancisco.png" 

#filename_geojson = "sfc_small.geojson"
#filename_geojson = "sfc_main.geojson"
filename_geojson = "SanFrancisco2.geojson"
#filename_geojson = "sfc.geojson"


number_of_cars = int(input("number_of_cars: "))
num_of_division = int(input("num_of_division: "))
epsilon = float(input("epsilon greedy: "))

sensitivity = 1.0

# functions
def read_parse_netxml(infilename):
  tree = ET.parse(infilename)
  root = tree.getroot()

  return root


def get_map_smopy():
  infile = open(filename_geojson, "r")
  data_dic = json.load(infile)

  max_lon = -180.0; min_lon = 180.0 
  max_lat = -90.0; min_lat = 90.0 
  for l in data_dic["features"][0]["geometry"]["coordinates"][0]:
    #print(l)
    if max_lon < float(l[0]):
      max_lon = float(l[0])
    if max_lat < float(l[1]):
      max_lat = float(l[1])
    if min_lon > float(l[0]):
      min_lon = float(l[0])
    if min_lat > float(l[1]):
      min_lat = float(l[1])

  lon_lat_tuple = (min_lat, min_lon, max_lat, max_lon)
  print(lon_lat_tuple)

  z=17
  smopy_map = smopy.Map(lon_lat_tuple, tileserver="https://tile.openstreetmap.org/{z}/{x}/{y}.png", tilesize=256, maxtiles=16, z=z)
  #smopy_map = smopy.Map(lon_lat_tuple, tileserver="http://a.tile.stamen.com/toner/{z}/{x}/{y}.png" ,tilesize=256, maxtiles=16, z=z)
  print("got map")

  px_min_lon, px_min_lat = smopy_map.to_pixels( lat=lon_lat_tuple[0], lon=lon_lat_tuple[1] )
  px_max_lon, px_max_lat = smopy_map.to_pixels( lat=lon_lat_tuple[2], lon=lon_lat_tuple[3] )

  x0 = min(px_max_lon, px_min_lon)
  x1 = max(px_max_lon, px_min_lon)
  y0 = min(px_max_lat, px_min_lat)
  y1 = max(px_max_lat, px_min_lat)

  smopy_map.save_png(png_infilename)

  return smopy_map, x0, x1, y0, y1, lon_lat_tuple


def latlng_to_xyz(lat, lng):
    rlat, rlng = radians(lat), radians(lng)
    coslat = cos(rlat)
    return coslat*cos(rlng), coslat*sin(rlng), sin(rlat)


def dist_on_sphere(pos0, pos1, radius=earth_rad):
    if pos0 == pos1:
        return 0
    xyz0, xyz1 = latlng_to_xyz(*pos0), latlng_to_xyz(*pos1)
    return acos(sum(x * y for x, y in zip(xyz0, xyz1)))*radius


def create_road_network(root, smopy_map):
  def get_boundary(root):
    for child in root:
      if child.tag == "location":
        convBoundary = list(map(float,child.attrib["convBoundary"].split(",")))
        origBoundary = list(map(float,child.attrib["origBoundary"].split(",")))
        #print(convBoundary)
        #print(origBoundary)
      return convBoundary, origBoundary

  def calculate_coordinates(convBoundary, origBoundary, node_x, node_y):
    orig_per_conv_X = abs(origBoundary[0] - origBoundary[2]) / abs(convBoundary[0] - convBoundary[2])
    orig_per_conv_Y = abs(origBoundary[1] - origBoundary[3]) / abs(convBoundary[1] - convBoundary[3])

    node_x = origBoundary[0] + (node_x * orig_per_conv_X)
    node_y = origBoundary[1] + (node_y * orig_per_conv_Y)
    return node_x, node_y

  def is_not_roadway(child):
    childs = str(child.attrib).split(",")
    for ch in childs:
      if "railway" in ch:
        return True
      if "highway.cycleway" in ch:
        return True
      if "highway.footway" in ch:
        return True
      if "highway.living_street" in ch:
        return True
      if "highway.path" in ch:
        return True
      if "highway.pedestrian" in ch:
        return True
      if "highway.step" in ch:
        return True
    return False


  # read edge tagged data for reading the road network
  # create data structure of road network using NetworkX
  x_y_dic = {} # input: node's x,y pos, output: node id
  DG = nx.DiGraph() # Directed graph of road network
  edge_lanes_list = [] # list of lane instances
  node_id = 0
  convBoundary, origBoundary = get_boundary(root)

  top = origBoundary[3]
  bottom = origBoundary[1]
  leftmost = origBoundary[0]
  rightmost = origBoundary[2]
  x_of_divided_area = abs(leftmost - rightmost) / num_of_division
  y_of_divided_area = abs(top - bottom) / num_of_division

  node_id_to_index = {}
  index_to_node_id = {}
  node_id_to_coordinate = {} # use for calculation distance

  for child in root:
    if child.tag == "edge":
      if is_not_roadway(child):
        continue

      lane = Lane()
      if "from" in child.attrib and "to" in child.attrib:
        lane.add_from_to(child.attrib["from"], child.attrib["to"])

      for child2 in child:

        try:
          data_list = child2.attrib["shape"].split(" ")
        except: # except param
          continue

        node_id_list = []
        node_x_list = []; node_y_list = []
        distance_list = []
        data_counter = 0

        for data in data_list:
          node_lon, node_lat = calculate_coordinates(convBoundary, origBoundary, float(data.split(",")[0]), float(data.split(",")[1]))

          node_x, node_y = smopy_map.to_pixels(node_lat,node_lon)


          index_x = int(abs(leftmost - node_lon) // x_of_divided_area)
          index_y = int(abs(top - node_lat) // y_of_divided_area)

          #緯度経度(xml,geojson)の誤差?によりindex==num_of_divisionとなる場合があるため、エリア内に収まるように調整する
          if not 0 <= index_x < num_of_division:
            if index_x >= num_of_division:
              index_x = num_of_division - 1
            else:
              index_x = 0
          if not 0 <= index_y < num_of_division:
            if index_y >= num_of_division:
              index_y = num_of_division - 1
            else:
              index_y = 0

          node_id_to_index[node_id] = (index_x, index_y)
          if (index_x, index_y) not in index_to_node_id.keys():
            index_to_node_id[(index_x, index_y)] = [node_id]
          else:
            index_to_node_id[(index_x, index_y)].append(node_id)

          node_id_to_coordinate[node_id] = {
            "longitude": node_lon,
            "latitude": node_lat
          }

          node_x_list.append( node_x )
          node_y_list.append( node_y )
          if (node_x, node_y) not in x_y_dic.keys():
            node_id_list.append(node_id)
            DG.add_node(node_id, pos=(node_x, node_y))
            x_y_dic[ (node_x, node_y) ] = node_id
            node_id += 1
          else:
            node_id_list.append( x_y_dic[ (node_x, node_y) ] )
          if data_counter >= 1:
            distance_list.append( np.sqrt( (node_x - old_node_x)**2 + (node_y - old_node_y)**2) )
          old_node_x = node_x
          old_node_y = node_y
          data_counter += 1
        for i in range(len(node_id_list)-1):
          DG.add_edge(node_id_list[i], node_id_list[i+1], weight=distance_list[i], color="black", speed=float(child2.attrib["speed"])) # calculate weight here
        if "from" in child.attrib and "to" in child.attrib:
          lane.set_others(float(child2.attrib["speed"]), node_id_list, node_x_list, node_y_list)
          edge_lanes_list.append(lane)  # to modify here



  # extract strongly connected components of DG
  scc = nx.strongly_connected_components(DG)
  largest_scc = True
  for c in sorted(scc, key=len, reverse=True):
    #print(c)
    if largest_scc == True:
      #print("largest:", c)
      largest_scc = False
    else:
      #print("others:", c)
      c_list = list(c)
      for i in range(len(c_list)):
        DG.remove_node(c_list[i])
        for lane in edge_lanes_list:
          if lane.node_id_list[0] == int(c_list[i]) or lane.node_id_list[1] == int(c_list[i]):
            edge_lanes_list.remove(lane)

  return x_y_dic, DG, edge_lanes_list, node_id_to_index, index_to_node_id, node_id_to_coordinate


# generate a list of road segments for U-turn
def create_road_segments(edge_lanes_list):
  road_segments_list = []
  for i in range(len(edge_lanes_list)-1):
    for j in range(i+1, len(edge_lanes_list)):
      if edge_lanes_list[i].from_id == edge_lanes_list[j].to_id and edge_lanes_list[i].to_id == edge_lanes_list[j].from_id:
        road_segments_list.append(RoadSegment(edge_lanes_list[i], edge_lanes_list[j]))
        break
  return road_segments_list

# randomly select Orign and Destination lanes (O&D are different)
def select_OD_lanes(edge_lanes_list):
  origin_lane_id = np.random.randint(len(edge_lanes_list))
  destination_lane_id = origin_lane_id
  while origin_lane_id == destination_lane_id:
    destination_lane_id = np.random.randint(len(edge_lanes_list))
  return origin_lane_id, destination_lane_id

def find_OD_node_ids(origin_lane_id, destination_lane_id, x_y_dic):
  origin_node_id = x_y_dic[ ( edge_lanes_list[origin_lane_id].node_x_list[0], edge_lanes_list[origin_lane_id].node_y_list[0] ) ]
  destination_node_id = x_y_dic[ ( edge_lanes_list[destination_lane_id].node_x_list[-1], edge_lanes_list[destination_lane_id].node_y_list[-1] ) ]
  return origin_node_id, destination_node_id

def get_node_id_to_lane_dic(edge_lanes_list):
  node_id_to_lane = {}
  for i, lane in enumerate(edge_lanes_list):
    for node_id in lane.node_id_list:
      node_id_to_lane[node_id] = i
  return node_id_to_lane


# For initializing animation settings
#@jit
def init():
  line.set_data([], [])
  title.set_text("Simulation step: 0")
  return line, title, 

# animation update
#@jit(parallel=True)
animation_count = 0
index_time = 0
def animate(time):

  global animation_count
  animation_count += 1
  if animation_count % 100 == 0:
    print("animation step: " + str(animation_count), datetime.datetime.now())

  global index_time 
  if animation_count % 1000 == 0: #1hあたりのstep数が1000の場合
    index_time += 1
    if index_time > 23:
      index_time = 0

  global cars_list

  xdata = []; ydata=[]
  xdata_ride = []; ydata_ride = []
  dest_xdata = []; dest_ydata = []
  dest_xdata_ride = []; dest_ydata_ride = []
  reward_sum = 0

  def choose_dest_node_at_random():
    dest_lane_id = np.random.randint(len(edge_lanes_list))
    dest_node_id = x_y_dic[ ( edge_lanes_list[dest_lane_id].node_x_list[-1], edge_lanes_list[dest_lane_id].node_y_list[-1] ) ]
    while car.orig_node_id == dest_node_id:
      dest_lane_id = np.random.randint(len(edge_lanes_list))
      dest_node_id = x_y_dic[ ( edge_lanes_list[dest_lane_id].node_x_list[-1], edge_lanes_list[dest_lane_id].node_y_list[-1] ) ]
    return dest_node_id


  #print(len(cars_list))
  new_cars_list = []
  remove_cars_list = []

  for car in cars_list:
    reward_sum += car.total_reward

    x_new, y_new, goal_arrived_flag = car.move(DG, edges_cars_dic, sensitivity) 
    # update x_new and y_new
    if car.ride_flag:
      xdata_ride.append(x_new)
      ydata_ride.append(y_new)
    else:
      xdata.append(x_new)
      ydata.append(y_new)
    
    lat = node_id_to_coordinate[car.dest_node_id]["latitude"]
    lon = node_id_to_coordinate[car.dest_node_id]["longitude"]
    dest_x, dest_y = smopy_map.to_pixels(lat, lon)
    if car.ride_flag:
      dest_xdata_ride.append(dest_x)
      dest_ydata_ride.append(dest_y)
    else:
      dest_xdata.append(dest_x)
      dest_ydata.append(dest_y)


    if car.goal_arrived == True:
      ride_flag = False

      #print("car arrived")
      index_x, index_y = node_id_to_index[car.dest_node_id]
      #car.experience[index_y][index_x]["count"] += 1
      orig_index_x, orig_index_y = node_id_to_index[car.orig_node_id]
      #car.experience[orig_index_y][orig_index_x]["step"] += car.num_of_elapsed_steps
      orig_node_id = car.dest_node_id

      if (index_x, index_y) not in car.experience.keys():
        car.experience[(index_x, index_y)] = {
          "reward": 0,
          "count": 0,
          "step": 0,
          "reward per step": 0,
        }
      if (orig_index_x, orig_index_y) in car.experience.keys():
        car.experience[(orig_index_x, orig_index_y)]["step"] += car.num_of_elapsed_steps

        step = car.experience[(orig_index_x, orig_index_y)]["step"]
        reward = car.experience[(orig_index_x, orig_index_y)]["reward"]
        car.experience[(orig_index_x, orig_index_y)]["reward per step"] = reward / step

      #print(car.experience)

      # 目的地の設定
      while True:
        if ride_prob[index_time][index_y][index_x] >= random.random() and len(reward_each_area[index_y][index_x]):
          #そのエリアで乗客を拾える場合
          ride_flag = True

          passenger_num_in_the_area = random.randrange(len(reward_each_area[index_y][index_x]))
          dest_x = reward_each_area[index_y][index_x][passenger_num_in_the_area]["index_x"]
          dest_y = reward_each_area[index_y][index_x][passenger_num_in_the_area]["index_y"]

          try:
            node_ids = index_to_node_id[(dest_x, dest_y)]
            index_of_node_ids = random.randrange(len(node_ids))
            dest_node_id = node_ids[index_of_node_ids]
          except KeyError:
            #サンフランシスコのタクシーのデータによる移動の目的地にノードが存在しない場合、そのデータを除外
            reward_each_area[index_y][index_x].pop(passenger_num_in_the_area)
            continue

        #epsilon greedy
        #乗客を拾えなかった場合、epsilon greedy で次の目的地を決める
        else:
          ride_flag = False
          if epsilon >= random.random(): #探索
            dest_node_id = choose_dest_node_at_random()

          #現在地からの距離を考慮した、過去の経験からの目的地の設定
          else: #活用
            max_reward_per_step = -float("inf")
            max_index = None
            if len(car.experience) == 0: #過去の経験がない場合はランダムで選ぶ
              dest_node_id = choose_dest_node_at_random()
            else:
              for i in car.experience[index_time]:
                for index, experience in i.items():
                  x_diff = abs(index_x - index[0])
                  y_diff = abs(index_y - index[1])
                  distance = np.sqrt((x_diff * X) ** 2 + (y_diff * Y) ** 2) #現在地から仮の目的地への距離
                  #print("distance", distance)
                  tmp_reward = -(distance / 10) #仮の目的地への移動コスト
                  if experience["count"] != 0:
                    step_per_count = experience["step"] / experience["count"]
                    #count_per_step = experience["count"] / experience["step"]
                    if step_per_count != 0: #count_per_step != 0:
                      tmp_reward /= step_per_count
                      #tmp_reward /= count_per_step #1stepあたりのrewardにかかる移動コスト
                    else:
                      tmp_reward = 0
                  else:
                    tmp_reward = 0
                else:
                  tmp_reward = 0

                tmp_reward_per_step = experience["reward per step"] + tmp_reward
                #print("#", experience["reward per step"], tmp_reward)
                if tmp_reward_per_step > max_reward_per_step:
                  max_reward_per_step = tmp_reward_per_step
                  max_index = index

              #print(max_index, max_reward_per_step)

              dest_x, dest_y = max_index
              node_ids = index_to_node_id[(dest_x, dest_y)]
              index_of_node_ids = random.randrange(len(node_ids))
              dest_node_id = node_ids[index_of_node_ids]

        break

    
      # calculation shortest path
      while True:
        try:
          shortest_path = nx.dijkstra_path(DG, orig_node_id, dest_node_id)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
          ride_flag = False #最短経路の計算でエラーの場合は、目的地をランダムで再設定
          dest_node_id = choose_dest_node_at_random()
          continue
        if len(shortest_path) == 1:
          dest_node_id = choose_dest_node_at_random()
          continue
        if ride_flag:
          car.experience[index_time][(index_x, index_y)]["reward"] += reward_each_area[index_y][index_x][passenger_num_in_the_area]["reward"]
          car.total_reward += reward_each_area[index_y][index_x][passenger_num_in_the_area]["reward"]
          #ride_flag = False
          #print("get reward")
        break
      
      #移動によるマイナス報酬の設定
      orig_pos = node_id_to_coordinate[orig_node_id]["latitude"], node_id_to_coordinate[orig_node_id]["longitude"]
      dest_pos = node_id_to_coordinate[dest_node_id]["latitude"], node_id_to_coordinate[dest_node_id]["longitude"]

      # append data for destination_coordinates_data?.?.txt
      car_id_datas.append(car)
      time_datas.append(time)
      orig_pos_datas.append(orig_pos)
      dest_pos_datas.append(dest_pos)

      moving_distance = dist_on_sphere(orig_pos, dest_pos)
      car.experience[(index_x, index_y)]["reward"] -= moving_distance / 10 # 10km/L 1L/1$
      car.total_reward -= moving_distance / 10
      
      car.experience[(index_x, index_y)]["count"] += 1

      # create new car
      new_car = Car(orig_node_id, dest_node_id, shortest_path, num_of_division)
      new_car.experience = car.experience
      new_car.max_reward_per_step_index = car.max_reward_per_step_index
      new_car.total_reward = car.total_reward
      new_car.init(DG)
      new_car.ride_flag = ride_flag
      edges_cars_dic[ ( new_car.shortest_path[ new_car.current_sp_index ], new_car.shortest_path[ new_car.current_sp_index+1] ) ].append( new_car )

      remove_cars_list.append(car)
      new_cars_list.append(new_car)

      del car
      gc.collect()

  for old_car in remove_cars_list:
    cars_list.remove(old_car)

  for new_car in new_cars_list:
    cars_list.append(new_car)



    # TODO: if the car encounters road closure, it U-turns.

  total_rewards.append(reward_sum / number_of_cars)

  # check if all the cars arrive at their destinations
  if len(cars_list) == 0:
    print("Total simulation step: "+str(time-1))
    print("### End of simulation ###")
    print("### Saving animation started ###")
    print("### Saving animation finished ###")
    sys.exit(0) # end of simulation, exit.

  line.set_data(xdata, ydata)
  ride.set_data(xdata_ride, ydata_ride)
  dest.set_data(dest_xdata, dest_ydata)
  dest_ride.set_data(dest_xdata_ride, dest_ydata_ride)
  title.set_text("Simulation step: "+str(time)+";  # of cars: "+str(len(cars_list)))

  return line, title, 


# Optimal Velocity Function
#@jit(nopython=True, parallel=True)
def V(b, current_max_speed):
  return 0.5*current_max_speed*(np.tanh(b-2) + np.tanh(2))


##### main #####
if __name__ == "__main__":

  smopy_map, x0, x1, y0, y1, lon_lat_tuple = get_map_smopy()
  # distance of devided area
  X = dist_on_sphere((lon_lat_tuple[0], lon_lat_tuple[1]), (lon_lat_tuple[0], lon_lat_tuple[3])) / num_of_division
  Y = dist_on_sphere((lon_lat_tuple[0], lon_lat_tuple[1]), (lon_lat_tuple[2], lon_lat_tuple[1])) / num_of_division

  ride_prob, reward_each_area = get_ride_prob_and_reward(infilename, num_of_division)

  # root: xml tree of input file 
  root = read_parse_netxml(infilename)

  # x_y_dic: node's x,y pos --> node id
  # DG: Directed graph of road network
  # edge_lanes_list: list of lane instances
  print("### create road network started ###", datetime.datetime.now())
  x_y_dic, DG, edge_lanes_list, node_id_to_index, index_to_node_id, node_id_to_coordinate = create_road_network(root, smopy_map)
  print("### create road network ended ###", datetime.datetime.now())
  
  node_id_to_lane = get_node_id_to_lane_dic(edge_lanes_list)

  # road_segments_list: list of road segment instances
  print("### create road segments started ###", datetime.datetime.now())
  road_segments_list = create_road_segments(edge_lanes_list)
  print("### create road segments ended ###", datetime.datetime.now())
  
  # create cars
  print("### create cars started ###", datetime.datetime.now())
  edges_all_list = DG.edges()
  edges_cars_dic = {}
  for item in edges_all_list:
    edges_cars_dic[ item ] = []
  cars_list = []

  len_shortest_path_list = []

  i = 0
  while i < number_of_cars:
    # randomly select Orign and Destination lanes (O&D are different)
    #print(len(edge_lanes_list))
    origin_lane_id, destination_lane_id = select_OD_lanes(edge_lanes_list)

    # find Orign and Destination node IDs
    origin_node_id, destination_node_id = find_OD_node_ids(origin_lane_id, destination_lane_id, x_y_dic)

    # calculate a shortest path to go
    # Reference: https://networkx.github.io/documentation/latest/reference/algorithms/generated/networkx.algorithms.shortest_paths.weighted.dijkstra_path.html
    #print("### calculate shortest path started ###")
    
    try:
      shortest_path = nx.dijkstra_path(DG, origin_node_id, destination_node_id)
    except nx.NetworkXNoPath:
      print("No Path")
      continue
    except nx.NodeNotFound:
      print("origin_node_id not in DiGraph")
      continue

    if len(shortest_path) == 1:
      continue
    #print("### calculate shortest path ended ###")

    car = Car(origin_node_id, destination_node_id, shortest_path, num_of_division)
    car.init(DG) # initialization of car settings
    cars_list.append(car)
    edges_cars_dic[ ( edge_lanes_list[origin_lane_id].node_id_list[0], edge_lanes_list[origin_lane_id].node_id_list[1] ) ].append( car )
    i += 1

  print("### create cars ended ###", datetime.datetime.now())
  
  # animation initial settings
  #fig, ax = plt.subplots()
  fig = plt.figure()
  print(x0,x1,y0,y1)
  ax = fig.add_subplot(111, autoscale_on=False, xlim=(x0,x1), ylim=(y0,y1)) 
  
  xdata = []; ydata = []
  
  for car in cars_list:
    xdata.append( car.current_position[0] )
    ydata.append( car.current_position[1] )
  line, = plt.plot([], [], color="green", marker="s", linestyle="", markersize=5)
  ride, = plt.plot([], [], color="blue", marker="s", linestyle="", markersize=5)
  dest, = plt.plot([], [], color="red", marker="*", linestyle="", markersize=6) #目的地のプロット
  dest_ride, = plt.plot([], [], color="blue", marker="*", linestyle="", markersize=6) #目的地のプロット
  title = ax.text(20.0, -20.0, "", va="center")

  
  print("### map image loading ###", datetime.datetime.now())
  img = Image.open(png_infilename)
  img_list = np.asarray(img)
  plt.imshow(img_list)
  print("### map image loaded ###", datetime.datetime.now())

  ax.invert_yaxis()

  gc.collect()

  print("### Start of simulation ###", datetime.datetime.now())
  total_rewards = []
  orig_pos_datas = []
  dest_pos_datas = []
  car_id_datas = []
  time_datas = []

  ani = FuncAnimation(fig, animate, frames=range(100000), init_func=init, blit=True, interval= 50)
  ani.save(str(epsilon) + "sfc-small.mp4", writer="ffmpeg")

  output.reward(total_rewards, epsilon)
  output.heatmap(cars_list, num_of_division, epsilon)

  with open("total_reward_" + str(epsilon) + ".txt", "w") as f:
    for reward in total_rewards:
      f.write(str(reward) + "\n")
  
  with open("experience_" + str(epsilon) + ".txt", "w") as f:
    for car in cars_list:
      for index, experience in car.experience.items():
        f.write(str(index) + str(experience) + "\n")

  with open("destination_coordinates_data" + str(epsilon) + ".txt", "w") as f:
    for car_id_data, time_data, orig_pos_data, dest_pos_data in zip(car_id_datas, time_datas, orig_pos_datas, dest_pos_datas):
      f.write(str(car_id_data) + "," + str(time_data) + "," + str(orig_pos_data) + "," + str(dest_pos_data) + "\n")

  for car in cars_list:
    print(car.experience)
  



