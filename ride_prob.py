import xml.etree.ElementTree as ET
import seaborn as sns
import matplotlib.pyplot as plt
from reward import reward_calculation

def read_parse_netxml(infilename):
    tree = ET.parse(infilename)
    root = tree.getroot()
    return root


def get_boundary(root):
    for child in root:
        if child.tag == "location":
            convBoundary = list(map(float,child.attrib["convBoundary"].split(",")))
            origBoundary = list(map(float,child.attrib["origBoundary"].split(",")))
        #print(convBoundary)
        #print(origBoundary)
        return convBoundary, origBoundary


def get_filepath_of_taxies(infilename):
    infilename_list = []
    with open(infilename,"r") as f:
        for line in f:
            line = line.rstrip().split('"')
            filepath = "./cabspottingdata/" + "new_" + line[1] + ".txt"
            infilename_list.append(filepath)
    return infilename_list


def extract_ride_point_reward(infilename_list):
    ride_points = []
    reward_list = []

    for infilename in infilename_list:
        #データの読み込み
        with open(infilename, "r") as f:
            #print(infilename)
            data_lists = []
            for line in f:
                data_tmp_list = line.split(" ")
                unixtime = int(data_tmp_list[3].replace('\r\n',''))
                longitude = float(data_tmp_list[1])
                latitude = float(data_tmp_list[0])
                ride_state = int(data_tmp_list[2]) # 1 or 0
                data_dict = {
                    "unixtime": unixtime,
                    "longitude": longitude,
                    "latitude": latitude,
                    "ride_state": ride_state
                }
                data_lists.append(data_dict)

        #unixtimeでソート
        data_lists = sorted(data_lists, key=lambda x:x["unixtime"])

        #乗車地点の抽出
        prev_ride_state = None
        coordinates_in_ride = []
        for data_dict in data_lists:
            ride_state = data_dict["ride_state"]
            if prev_ride_state is None:
                prev_ride_state = ride_state
                if ride_state == 1:
                    ride_points.append(data_dict)
                    departure_time = data_dict["unixtime"]
                continue

            if not (prev_ride_state == 0 and ride_state == 0):
                coordinates_in_ride.append(data_dict)
                if prev_ride_state == 0 and ride_state == 1:
                    ride_points.append(data_dict)
                    departure_time = data_dict["unixtime"]

                elif prev_ride_state == 1 and ride_state == 0:
                    arrival_time = data_dict["unixtime"]

                    reward, orig, dist = reward_calculation(coordinates_in_ride)
                    elapsed_time = arrival_time - departure_time
                    if reward is not None and 2400 > elapsed_time > 120 and reward < 40: #simulationを行うエリアの大きさによって変える必要有
                        reward_list.append([reward, orig, dist, elapsed_time]) 
                    else:
                        #print("error 2min")
                        ride_points.pop()
                    coordinates_in_ride = []

            prev_ride_state = ride_state

    return ride_points, reward_list


def find_ride_num_reward_each_area(num_of_division, origBoundary, ride_points, reward_list):
    top = origBoundary[3]
    bottom = origBoundary[1]
    leftmost = origBoundary[0]
    rightmost = origBoundary[2]

    #分割されたエリアの辺の長さ　単位は緯度、経度
    x_of_divided_area = abs(leftmost - rightmost) / num_of_division
    y_of_divided_area = abs(top - bottom) / num_of_division

    #[0][0]左上 [max][max]右下　エリアごとの乗客数を入れるリスト
    ride_num_each_area = [[[0 for i in range(num_of_division)] for j in range(num_of_division)] for k in range(24)]
    reward_each_area = [[[] for i in range(num_of_division)] for j in range(num_of_division)]

    for data_dict in ride_points:
        longitude = data_dict["longitude"] #経度
        latitude = data_dict["latitude"] #緯度
        index_x = int(abs(leftmost - longitude) // x_of_divided_area)
        index_y = int(abs(top - latitude) // y_of_divided_area)
        if 0 <= index_x < num_of_division - 1 and 0 <= index_y < num_of_division - 1:
            ride_num_each_area[index_y][index_x] += 1

    for reward, orig, dist, elapsad_time in reward_list:
        longitude = orig[1]
        latitude = orig[0]
        orig_x = int(abs(leftmost - longitude) // x_of_divided_area)
        orig_y = int(abs(top - latitude) // y_of_divided_area)
        dist_x = int(abs(leftmost - dist[1]) // x_of_divided_area)
        dist_y = int(abs(top - dist[0]) // y_of_divided_area)
        if 0 <= orig_x < num_of_division - 1 and 0 <= orig_y < num_of_division - 1:
            if 0 <= dist_x < num_of_division - 1 and 0 <= dist_y < num_of_division - 1:
                #dist_x = int(abs(leftmost - dist[1]) // x_of_divided_area)
                #dist_y = int(abs(top - dist[0]) // y_of_divided_area)

                tmp = {
                    "reward": reward,
                    "index_x": dist_x,
                    "index_y": dist_y,
                    "elapesed_time": elapsad_time
                }
                reward_each_area[orig_y][orig_x].append(tmp)
    
    return ride_num_each_area, reward_each_area


def find_ride_prob(ride_num_each_area):
    num_of_division = len(ride_num_each_area)

    max_ride_num = 0
    for i in range(num_of_division):
        for j in range(num_of_division):
            if max_ride_num < ride_num_each_area[i][j]:
                max_ride_num = ride_num_each_area[i][j]
    prob_increase_per_a_ride = 1 / max_ride_num
    #prob_increase_per_a_ride = 0.8 / max_ride_num


    ride_prob = [[0 for i in range(num_of_division)] for j in range(num_of_division)]
    for i in range(num_of_division):
        for j in range(num_of_division):
            ride_prob[i][j] = ride_num_each_area[i][j] * prob_increase_per_a_ride

    return ride_prob


def get_ride_prob_and_reward(filename_of_xml, num_of_division):
    file_of_taxi = "./cabspottingdata/_cabs.txt"

    root = read_parse_netxml(filename_of_xml)
    _, origBoundary = get_boundary(root)
    infilename_taxies = get_filepath_of_taxies(file_of_taxi)
    ride_points, reward_list = extract_ride_point_reward(infilename_taxies)
    ride_num_each_area, reward_each_area = find_ride_num_reward_each_area(num_of_division, origBoundary, ride_points, reward_list)
    ride_prob = find_ride_prob(ride_num_each_area)
    return ride_prob, reward_each_area


def main():
    filename_of_xml = "SanFrancisco2.net.xml"
    file_of_taxi = "./cabspottingdata/_cabs.txt"

    #1辺を何分割するか
    num_of_division = int(input("num_of_division: "))

    root = read_parse_netxml(filename_of_xml)
    print("///read netxml")
    _, origBoundary = get_boundary(root)
    print("///got boundary")

    infilename_taxies = get_filepath_of_taxies(file_of_taxi)
    #infilename_taxies = ["./cabspottingdata/" + "tmp" + ".txt"]
    print("///got filepath of taxies")
    ride_points, reward_list = extract_ride_point_reward(infilename_taxies)
    print("///extracted ride points and reward")
    ride_num_each_area, reward_each_area = find_ride_num_reward_each_area(num_of_division, origBoundary, ride_points, reward_list)
    print("///counted ride num each area")
    #print(ride_num_each_area)
    ride_prob = find_ride_prob(ride_num_each_area)
    print("///calculated ride probability")
    #print(ride_prob)

    #print(reward_each_area[8][10])
    #cnt = 0
    #err_cnt = 0
    #for i in range(num_of_division):
    #    for j in range(num_of_division):
    #        for dic in reward_each_area[i][j]:
    #            cnt += 1
    #            if dic["elapesed_time"] > 2400:
    #                err_cnt += 1
    #                print(dic)
    #            if dic["reward"] > 40:
    #               err_cnt += 1
    #                print(dic)

    #print("cnt " + str(cnt))
    #print("err_cnt " + str(err_cnt))
    #for dic in reward_each_area[8][10]:
    #    if dic["elapesed_time"] > 3600:
    #        print(dic)
    #    if dic["reward"] > 100:
    #        print(dic)
    #print()
    #print(reward_each_area[2][3])
    #print()
    fig = plt.figure(dpi=300)
    ax1 = fig.add_subplot(111)
    sns.heatmap(ride_prob, cmap='coolwarm', square=True, robust=True, ax=ax1)
    plt.xticks(color="None")
    plt.yticks(color="None")
    plt.tick_params(length=0)
    plt.savefig("ride_probability_distribution.png")
    #plt.show()


if __name__ == "__main__":
    main()