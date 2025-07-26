class Lane:
  def __init__(self):
    self.car_list = []

  def add_from_to(self, from_id, to_id):
    self.from_id = from_id
    self.to_id = to_id

  def set_others(self, speed, node_id_list, node_x_list, node_y_list):
    self.speed = speed
    self.node_id_list = node_id_list
    self.node_x_list = node_x_list
    self.node_y_list = node_y_list

