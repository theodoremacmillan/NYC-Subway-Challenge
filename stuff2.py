import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np
from mpl_toolkits import mplot3d

class Node:
    def __init__(self,lon,lat,id,neighbors,name):
        self.lon = lon
        self.lat = lat
        self.id = id
        self.neighbors = neighbors
        self.name = name

def seconds(time):
    h, m, s = map(int, time.split(':'))
    return h * 60**2 + m * 60 + s

filename = "google_transit/stop_times.txt"
f = open(filename, 'r')

full_list = []

for line in f:
    trip_id = line.split(',')[0].split('-')

    full_list.append(line)

full_list = full_list[1:]

full_list[1]


full_list[0].split(",")
name_list = []
time_list = []
current_trip_id = ""

timed_graph = {}
graph = {}

trips = []

last_trip = ""
last_name = ""
last_time = ""
for item in full_list:
    split_list = item.split(",")

    trip = split_list[0]
    name = split_list[3]
    time = seconds(split_list[1])
    
    if name not in graph:
        graph[name] = []
    if (name, time) not in timed_graph:
        timed_graph[(name, time)] = []

    if trip == last_trip:
        timed_graph[(last_name, last_time)].append((name, time))
        if name not in graph[last_name]:
            graph[last_name].append(name)
    else:
        trips.append(trip)

    last_trip = trip
    last_name = name
    last_time = time

print(trips)

print(f"{len(timed_graph) = }")
print(f"{len(graph) = }")
edges = 0
for name in graph:
    edges += len(graph[name])
print(f"{edges = }")
print(' '.join(sorted(list(graph))))


filename = "google_transit/transfers.txt"
f = open(filename, 'r')

trans_graph = {}

graph_size = 0
count = 0
for line in f:
    count += 1
    if count == 1:
        continue
    info = line.split(',')
    stats = []
    stats.append(info[0] + 'N')
    stats.append(info[0] + 'S')
    stats.append(info[1] + 'N')
    stats.append(info[1] + 'S')
    stats = list(set(stats))
    time = int(info[3])

    for item1 in stats:
        for item2 in stats:
            if item1 == item2:
                continue
            if item1 not in trans_graph:
                trans_graph[item1] = {}
            if item2 not in trans_graph[item1]:
                trans_graph[item1][item2] = 60**3
                trans_graph[item1][item2] = min(trans_graph[item1][item2], time)
            graph_size += 1

#for stat in trans_graph:
#    neighs = trans_graph[stat]
#    if len(neighs) > 1:
#        print(f"{stat} : {len(neighs)}")
print(len(trans_graph))

for name in graph:
    if name not in trans_graph:
        continue
    for neigh in trans_graph[name]:
        if neigh not in graph[name]:
            graph[name].append(neigh)

for name in graph:
    #print(f"{name} : {graph[name]}")
    edges += len(graph[name])
print(f"{edges = }")

new_times = []

for name, time in timed_graph:
    if name not in trans_graph:
        continue
    trans = trans_graph[name]
    for neigh in trans:
        time2 = time + trans[neigh]
        neigh_time = (neigh, time2)
        timed_graph[(name, time)].append(neigh_time)
        if neigh_time not in timed_graph:
            new_times.append(neigh_time)
        graph[name].append(neigh)

for neigh_time in new_times:
    timed_graph[neigh_time] = []

print(len(timed_graph))
