import pandas as pd
import csv
import json
import math
from datetime import datetime
import matplotlib.pyplot as plt

data = pd.read_csv('data.csv')
orders_by_size = [10263914, 10271810, 10268285, 10264218, 10276617, 10268087, 10276921, 10279434, 10279304, 10265550, 10273470, 10271728, 10278741, 10279182, 10272558, 10274460, 10265413, 10271858, 10274453, 10275146, 10275283, 10268483, 10268872, 10272657, 10272664, 18120226, 10277126, 10277577, 10268780, 10268797, 10269190, 10268063, 10273487, 10275863, 10271827, 10272572, 10272619, 10274552, 19477244, 10273555, 10275245, 10277157, 10268919, 10271599, 10274088, 10275153, 10278765, 10278758, 10265567, 10268261, 10268902, 10272756, 10274569, 10268988, 10273517, 10274071, 10279236, 10268506, 10268896, 10269183, 10271841, 10271711, 10272718, 10278185, 10271834, 10268056, 10271742, 10275252, 10268841, 10277140, 10269213, 10271261, 10271629, 10272701, 10272732, 10279175, 10268070, 10273548, 10279243, 10279212, 10271568, 10272787, 10272589, 10275214, 10276907, 10279229, 10271582, 19476902, 10274095, 10275870, 10275269, 10269152, 18120172, 10272688, 10272770, 10273395, 18120196, 10275191, 10278932, 10279205, 10269169, 10268889, 10271537, 10272763, 10269206, 10271575, 10271650, 10271667, 10271544, 10271612, 18120219, 19477060, 10271278, 10271643, 10271551, 10271636, 10271605, 10279199, 10272626, 10272749, 10269176, 10268865, 19476810, 10272671, 19476476, 10273524, 10273494, 10269220, 10272695, 10268049, 10273371, 10272640, 18120165, 10273388, 10279250, 10273364, 10271674, 10267974, 18120301]

def average_difference(order_id,coordinate):
    '''
    this function checks the average difference between the agent's tagging and AIs tagging.
    :param order_id: int
    :param coordinate: x,y,height,width
    :return: int
    '''
    order_details = data.loc[data['order_id']==order_id,["ai_tracker_last_bbox","agent_last_bbox"]]
    diff = 0
    for (ai,agent) in zip(order_details['ai_tracker_last_bbox'].dropna(),order_details['agent_last_bbox'].dropna()):
        diff += math.sqrt((json.loads(ai)[coordinate]-json.loads(agent)[coordinate])**2)

    return diff/len(order_details['ai_tracker_last_bbox'].dropna())

def all_average_difference(data,coordiante):
    total_distance = 0
    max_time = 0
    max_id = ''
    min_time = math.inf
    min_id = ''
    for order in lst:
        distance = average_difference(order,coordiante)
        total_distance += distance
        if distance > max_time:
            max_time = round(distance,2)
            max_id = order
        elif distance < min_time:
            min_time = round(distance,2)
            min_id = order
    return f'The average distance for the {coordiante} axis is {round(total_distance/len(lst),2)}.\nThe order with the biggest difference is {max_id} with {max_time} differnce, and the order with the smallest difference is {min_id} with {min_time} difference'

def ai_detection_accuracy(order_id):
    '''
    checks the amount of times the AI worked with a specific order id
    :param order_id: int
    :return: float
    '''
    order_details = data.loc[data['order_id']==order_id,'ai_tracker_picker_found']
    tr = 0
    fa = 0
    for case in order_details:
        if case == True:
            tr += 1
        elif case == False:
            fa += 1
    return round(tr/(tr+fa)*100,3)

def all_detection_accuracy(data):
    max_acc = 0
    max_id = ''
    min_acc = math.inf
    min_id = ''
    total_accuracy = 0
    for order in data:

        accuracy = ai_detection_accuracy(order)
        total_accuracy += accuracy
        if accuracy > max_acc:
            max_acc = round(accuracy, 2)
            max_id = order
        elif accuracy < min_acc:
            min_acc = round(accuracy, 2)
            min_id = order
    return f'The average accuracy for all orders is {round(total_accuracy/len(data),2)}.\nThe order with the highest accuracy is {max_id} with {max_acc}%, and the order with the lowest accuracy is {min_id} with {min_acc}%'

def check_camera_accuracy(cam_id):
    '''
    checks a certain camera's accuracy
    :param cam_id: int
    :return: float
    '''
    cameras = data.loc[data['camera_id']==cam_id,'ai_tracker_picker_found']
    tr = 0
    fa = 0
    for cam in cameras:
        if cam == True:
            tr += 1
        elif cam == False:
            fa += 1
    return round(tr / (tr + fa) * 100, 2)


def camera_accuracy():
    '''
    uses func check_camera_accuracy for all cameras, returning the best and worst cameras
    :return:
    '''
    cameras = set(data.loc[data['camera_id'],'camera_id'])
    bestcam = -1
    bestscore = 0
    worstcam = -1
    worstscore = 100
    total_score = 0
    instance = 0
    for camera in cameras:
        score =check_camera_accuracy(camera)
        total_score += score
        instance += 1
        if  score > bestscore:
            bestscore = check_camera_accuracy(camera)
            bestcam = camera
        elif score < worstscore:
            worstscore = score
            worstcam = camera
    return f"The average score for all cameras is {round(total_score/instance,2)}.\nThe best camera is {bestcam} with a score of {bestscore}, and the worst camera is {worstcam} with a score of {worstscore}"


# Check number of misses (no. of agent tagged stalls vs ai tagged stalls)
def check_camera_misses(order_id):
    order_details = data.loc[data['order_id'] == order_id, ['ai_tracker_stalls', 'agent_stalls', 'camera_id']]
    agent_time = order_details.dropna()['agent_stalls']
    ai_time = order_details.dropna()['ai_tracker_stalls']
    camera_id = order_details.dropna()['camera_id']
    d = dict()
    for (ai, agent, cam) in zip(agent_time.dropna(), ai_time.dropna(), camera_id.dropna()):
        if cam in d:
            d[cam]["ai_count"] += len(json.loads(ai))
            d[cam]["agent_count"] += len(json.loads(agent))
        else:
            d[cam] = {"ai_count": len(json.loads(ai)), "agent_count": len(json.loads(agent))}
    problematic_cameras = []
    for camera in d:
        if d[camera]['agent_count'] - d[camera]['ai_count'] > 1:
            problematic_cameras.append(camera)

    return problematic_cameras

# Average time differences of stalls
def find_time_differences(ts1,ts2):
    '''
    helper function
    recieves two strings in formart str: "YYYY-MM-DD HH:MM:SS.XXXXXX+XX:XX" and returns the time differnce between them.
    :param ts1: str
    :param ts2: str
    :return: float
    '''
    timestamp1 = datetime.fromisoformat(ts1)
    timestamp2 = datetime.fromisoformat(ts2)
    return (timestamp2 - timestamp1).total_seconds()

def stall_time_diff():
    order_details = data[['ai_tracker_stalls', 'agent_stalls']]
    agent_time = order_details['agent_stalls'].dropna()
    ai_time = order_details['ai_tracker_stalls'].dropna()
    total_difference = 0
    instances = 0
    max_diff = 0
    min_diff = math.inf
    for (ai,agent) in zip(ai_time,agent_time):
        ai = json.loads(ai)
        agent = json.loads(agent)
        if len(ai) == len(agent):
            for (event1,event2) in zip(ai,agent):
                cur_dif = abs(find_time_differences(event1['time'],event2['time']))
                if cur_dif > max_diff:
                    max_diff = cur_dif
                if cur_dif < min_diff:
                    min_diff = cur_dif
                total_difference += cur_dif
                instances += 1
    return f"The average time difference is {round(total_difference/instances,2)} seconds.\nThe maximum difference is {max_diff} seconds, and the min difference is {min_diff} seconds"


# Agent last time vs ai last time
def agent_vs_ai_lasttime(order_id):
    ai_lasttime = data.loc[data['order_id']==order_id,'ai_tracker_last_time']
    agent_lasttime = data.loc[data['order_id']==order_id,'agent_last_time']
    instances = 0
    time = 0
    for ai_time,agent_time in zip(ai_lasttime.dropna(),agent_lasttime.dropna()):
        # ai_time = json.loads(ai_time)
        # agent_time = json.loads(agent_time)
        diff = abs(find_time_differences(ai_time,agent_time))
        time += diff
        instances+=1
    return time/instances
# print(agent_vs_ai_lasttime(10264218))

def all_lasttime_differences(data):
    max_time = 0
    max_id = ''
    min_time = math.inf
    min_id = ''
    total_time = 0
    for order in data:
        time = agent_vs_ai_lasttime(order)
        total_time += time
        if time > max_time:
            max_time = round(time, 2)
            max_id = order
        elif time < min_time:
            min_time = round(time, 2)
            min_id = order
    return f'The average time difference for all orders is {round(total_time / len(data), 2)}.\nThe order with the largest time difference is {max_id} with {max_time} seconds, and the order with the lowest time difference is {min_id} with {min_time} seconds'

# Helper function - find the 3 biggest orders
def find_biggest_orders():
    orders = data['order_id']
    d = dict()

    for line in orders:
        if line in d:
            d[line] += 1
        else:
            d[line] = 1
    top_orders = sorted(d.items(), key=lambda x: x[1], reverse=True)[:3]
    return top_orders

def find_xandy_points(order_df):
    ai_coords = []
    agent_coords = []
    for ai_line,agent_line in zip(order_df['ai_tracker_last_bbox'].dropna(),order_df['agent_last_bbox']):
        ai_line = json.loads(ai_line)
        agent_line = json.loads(agent_line)
        ai_coords.append((ai_line['x'],ai_line['y']))
        agent_coords.append((agent_line['x'],agent_line['y']))

    return(ai_coords,agent_coords)

