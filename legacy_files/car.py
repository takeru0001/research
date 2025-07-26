import networkx as nx
import numpy as np
import math
import random

class Car:
  def __init__(self, orig_node_id, dest_node_id, shortest_path, num_of_division):
    self.orig_node_id  = orig_node_id
    self.experience = {}
    """
    car.experience[(index_x, index_y)] = {
      "reward": 0,
      "count": 0,
      "step": 0,
      "reward per step": 0,
    }
    """
    self.dest_node_id  = dest_node_id
    self.shortest_path = shortest_path
    self.current_sp_index = 0
    self.current_speed = 0.0
    self.current_start_node = []
    self.current_position = []
    self.current_end_node = []
    self.current_distance = 0.0
    self.goal_arrived = False

    self.num_of_elapsed_steps = 0
    self.max_reward_per_step_index = None
    self.total_reward = 0
    self.ride_flag = False

  def init(self, DG):
    current_start_node_id = self.shortest_path[ self.current_sp_index ]
    self.current_start_node = DG.nodes[ current_start_node_id ]["pos"]
    self.current_position = DG.nodes[ current_start_node_id ]["pos"]
    current_end_node_id = self.shortest_path[ self.current_sp_index+1]
    self.current_end_node = DG.nodes[ current_end_node_id ]["pos"]
    current_edge_attributes = DG.get_edge_data(current_start_node_id, current_end_node_id)
    self.current_max_speed = current_edge_attributes["speed"]
    self.current_distance = current_edge_attributes["weight"]


  # Optimal Velocity Function to determine the current speed
  def V(self, inter_car_distance):
    return 0.5*self.current_max_speed*(np.tanh(inter_car_distance-2) + np.tanh(2))

  # update car's speed
  def update_current_speed(self, sensitivity, inter_car_distance):
    self.current_speed += sensitivity*( self.V(inter_car_distance) - self.current_speed )

  def move(self, DG, edges_cars_dic, sensitivity):
    self.num_of_elapsed_steps += 1

    # x_prev == self.current_position[0]
    # y_prev == self.current_position[1]
    direction_x = self.current_end_node[0] - self.current_position[0]
    direction_y = self.current_end_node[1] - self.current_position[1]
    arg = math.atan2(direction_y, direction_x)

    arrived_cars_list = []
    #x_new = None; y_new = None

    if np.sqrt((self.current_position[0] - self.current_end_node[0])**2 + (self.current_position[1] - self.current_end_node[1])**2) < self.current_speed: # to arrive at the terminal of edge

      self.current_sp_index += 1

      if self.current_sp_index >= len(self.shortest_path)-1: # arrived at the goal
        self.goal_arrived = True
        x_new = self.current_end_node[0]
        y_new = self.current_end_node[1]

        current_start_node_id = self.shortest_path[ self.current_sp_index-1 ]
        current_end_node_id = self.shortest_path[ self.current_sp_index ]
        edges_cars_dic[ (current_start_node_id, current_end_node_id) ].remove( self )
        arrived_cars_list.append( self )

      else: # lane change
        x_new = self.current_end_node[0]
        y_new = self.current_end_node[1]

        current_start_node_id = self.shortest_path[ self.current_sp_index-1 ]
        current_end_node_id = self.shortest_path[ self.current_sp_index ]
        #print(edges_cars_dic)
        edges_cars_dic[ (current_start_node_id, current_end_node_id) ].remove( self )

        current_start_node_id = self.shortest_path[ self.current_sp_index ]
        self.current_start_node = DG.nodes[ current_start_node_id ]["pos"]
        self.current_position = DG.nodes[ current_start_node_id ]["pos"]
        current_end_node_id = self.shortest_path[ self.current_sp_index+1]
        self.current_end_node = DG.nodes[ current_end_node_id ]["pos"]
        current_edge_attributes = DG.get_edge_data(current_start_node_id, current_end_node_id)
        self.current_max_speed = current_edge_attributes["speed"]
        self.current_distance = current_edge_attributes["weight"]
        edges_cars_dic[ (current_start_node_id, current_end_node_id) ].append( self )
    else: # move to the terminal of edge
      x_new = self.current_position[0] + self.current_speed*np.cos(arg)
      y_new = self.current_position[1] + self.current_speed*np.sin(arg)
      self.current_position = [x_new, y_new]
      current_start_node_id = self.shortest_path[ self.current_sp_index ]
      current_end_node_id = self.shortest_path[ self.current_sp_index+1]
      #print(edges_cars_dic)
      #print(current_start_node_id, current_end_node_id)
      #print("fdsa")
      #print(edges_cars_dic[ (current_start_node_id, current_end_node_id) ].index( self ))
      if edges_cars_dic[ (current_start_node_id, current_end_node_id) ].index( self ) > 0:
        car_forward_index = edges_cars_dic[ (current_start_node_id, current_end_node_id) ].index( self ) - 1
        car_forward_pt = edges_cars_dic[ (current_start_node_id, current_end_node_id) ][ car_forward_index ]
        diff_dist = np.sqrt( (car_forward_pt.current_position[0] - self.current_position[0])**2 + (car_forward_pt.current_position[1] - self.current_position[1])**2 )
      else:
        diff_dist = 50.0
      #print(self, diff_dist)
      self.update_current_speed(sensitivity, diff_dist)

    return x_new, y_new, self.goal_arrived

