import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import hvplot.pandas
from collections import defaultdict
import time

def raptor(route_to_trips, transfers_df, stops_df, p_s, tau, K):
    # get a list of all stations
    unique_stop_ids = set([stop[:3] for stop in stops_df.stop_id])

    # each station has a K-tuple (multilabel) where tau_i(p) represents the earliest known arrival time at p with up to i transfers
    # initialized to a date very far in the future
    p_multilabel = {stop:[pd.to_timedelta('100:00:00') for i in range(K)] for stop in unique_stop_ids}

    # Round 0: Initialize the starting station
    for i in range(K):
        p_multilabel[p_s][i] = tau

    journeys = {stop:{'transfer_steps':[]} for stop in unique_stop_ids}
    journeys = {stop:[] for stop in unique_stop_ids}

    # run the steps of RAPTOR 1,...,K
    for k in range(1,K):
        # iterate once over each route
        for unique_route in route_to_trips.keys():
            for trip in route_to_trips[unique_route]:
                hopped_on = False
                intermediate_trip = []
                for (stop, time) in trip:
                    if hopped_on == True:
                        # now we've "hopped on" the train
                        intermediate_trip.append((stop,time))
                        if time < p_multilabel[stop][k]:
                            journeys[stop].append({'route':unique_route,
                                                    'starting_stop':trip_pointer[0],
                                                    'starting_time':trip_pointer[1],
                                                    'ending_stop':stop,
                                                    'ending_time':time,
                                                    'round':k,
                                                    'intermediate_stations':intermediate_trip.copy()})
                            for i in range(k,K):
                                p_multilabel[stop][k] = time

                    # if the departure time is after the earliest possible time you could 
                    # get to the station with k-1 transfers, lets hop on
                    if time > p_multilabel[stop][k-1]:
                        hopped_on = True
                        trip_pointer = (stop, time)

        # now we need to look at all the transfers
        for index, row in transfers_df.iterrows():
            from_stop = row.from_stop_id
            to_stop = row.to_stop_id
            transfer_time = pd.to_timedelta(row.min_transfer_time, unit='s')

            if p_multilabel[to_stop][k] > p_multilabel[from_stop][k] + transfer_time:
                p_multilabel[to_stop][k] = p_multilabel[from_stop][k] + transfer_time
                journeys[to_stop].append({
                                                    'route':'walking_transfer',
                                                    'starting_stop':from_stop,
                                                    'starting_time':p_multilabel[from_stop][k],
                                                    'ending_stop':to_stop,
                                                    'ending_time':p_multilabel[from_stop][k] + transfer_time,
                                                    'round':k})
            
            if p_multilabel[from_stop][k] > p_multilabel[to_stop][k] + transfer_time:
                p_multilabel[from_stop][k] = p_multilabel[to_stop][k] + transfer_time
                journeys[from_stop].append({
                                                    'route':'walking_transfer',
                                                    'starting_stop':to_stop,
                                                    'starting_time':p_multilabel[to_stop][k],
                                                    'ending_stop':from_stop,
                                                    'ending_time':p_multilabel[to_stop][k] + transfer_time,
                                                    'round':k})
                
    # now clean up the journeys
    fastest_journeys = {stop:[] for stop in unique_stop_ids}

    for p_t in unique_stop_ids:
        target = p_t

        fastest_journey = []

        # find the earliest arrival time to p_t
        while p_t != p_s:
            min_time = pd.to_timedelta('100:00:00')
            for trip in journeys[p_t]:
                if trip['ending_time'] < min_time:
                    fastest_arrival_trip = trip
                    min_time = trip['ending_time']

            fastest_arrival_trip
            # print(
            #     f'''Take the {fastest_arrival_trip["route"]} from {fastest_arrival_trip["starting_stop"]} to {fastest_arrival_trip["ending_stop"]}
            #     starting at {fastest_arrival_trip["starting_time"]} and ending at {fastest_arrival_trip["ending_time"]}''')
            
            fastest_journey.append(fastest_arrival_trip)
            p_t = fastest_arrival_trip['starting_stop']

        fastest_journeys[target] = fastest_journey[::-1]

    return fastest_journeys

# gets unique stops in a journey
def get_unique_journey_stops(journey):
    journey_stops = set()

    for trip in journey:
        journey_stops.add(trip['starting_stop'])
        journey_stops.add(trip['ending_stop'])

        if 'intermediate_stations' in trip:
            for stop in trip['intermediate_stations']:
                journey_stops.add(stop[0])

    return journey_stops

class VELOCIRAPTOR():
    def __init__(self, stop_times_df, transfers_df):
        self.stop_times_df = stop_times_df.copy()
        self.unique_stops = stop_times_df['stop_id'].unique()
        self.transfers_df = transfers_df.copy()

        # convert time strings to seconds
        self.stop_times_df['arrival_time'] = pd.to_timedelta(stop_times_df['arrival_time']).dt.total_seconds().astype(int)
        self.stop_times_df['departure_time'] = pd.to_timedelta(stop_times_df['departure_time']).dt.total_seconds().astype(int)

        shifted_stop_times = self.stop_times_df.copy()
        shifted_stop_times['trip_id'] = shifted_stop_times['trip_id'] + '_shifted'
        shifted_stop_times['arrival_time'] = shifted_stop_times['arrival_time'] + 60*60*24
        shifted_stop_times['departure_time'] = shifted_stop_times['departure_time'] + 60*60*24

        self.stop_times_df = pd.concat([self.stop_times_df, shifted_stop_times])

        # initialize datastructures
        self.extract_routes_and_trips()
        self.get_route_trips_dict()
        self.generate_footpaths_dict()


    def extract_routes_and_trips(self):
        df = self.stop_times_df
        # First, extract all unique trip sequences
        df_sorted = df.sort_values(['trip_id', 'arrival_time'])
        
        # Store trips with their stop-time pairs
        trips = {}
        # Store just the stop sequences for route identification
        trip_stop_sequences = {}
        
        for trip_id, group in df_sorted.groupby('trip_id'):
            # Store the full trip information (stops and times)
            trips[trip_id] = list(zip(group['stop_id'], group['arrival_time']))
            # Store just the stop sequence for route identification
            trip_stop_sequences[trip_id] = tuple(group['stop_id'])
        
        # Identify unique routes
        routes = {}
        trip_to_route = {}
        current_route_id = 1
        
        # Create reverse lookup of stop sequences to route_ids
        sequence_to_route = {}
        
        for trip_id, stop_sequence in trip_stop_sequences.items():
            if stop_sequence in sequence_to_route:
                # This sequence of stops already exists as a route
                route_id = sequence_to_route[stop_sequence]
            else:
                # This is a new route
                route_id = current_route_id
                routes[route_id] = list(stop_sequence)
                sequence_to_route[stop_sequence] = route_id
                current_route_id += 1
            
            trip_to_route[trip_id] = route_id

        self.routes = routes
        self.trips = trips
        self.trip_to_route = trip_to_route
        
        return routes, trips, trip_to_route
    
    def get_trips_for_route(self, route_id):
        route_trips = {}
        for trip_id, rid in self.trip_to_route.items():
            if rid == route_id:
                route_trips[trip_id] = self.trips[trip_id]
        return route_trips
    
    def get_route_trips_dict(self):
        self.route_trips_dict = {}
        for route_id in self.routes.keys():
            self.route_trips_dict[route_id] = self.get_trips_for_route(route_id)
    
    def get_earliest_trip(self, p_s, route_id, route_trips):
        # find the list index of the stop
        stop_idx = self.routes[route_id].index(p_s[0])
        earliest_trip = None
        earliest_time = np.inf

        for trip_id, stops in route_trips.items():
            # if the last stop of the trip is before tau, ignore the trip
            if stops[-1][1] < p_s[1]:
                continue

            if stops[stop_idx][1] < earliest_time and stops[stop_idx][1] >= p_s[1]:
                earliest_time = stops[stop_idx][1]
                earliest_trip = trip_id

        #print(earliest_trip, earliest_time)
        return earliest_trip
    
    def generate_footpaths_dict(self):
        self.footpaths = defaultdict(list)

        for stop in self.unique_stops:
            possible_transfers = self.transfers_df[self.transfers_df['from_stop_id'] == stop[:3]]

            # if there are no possible transfers, allow a self transfer with travel time 0
            if len(possible_transfers) == 0:
                possible_transfers = pd.DataFrame({'from_stop_id': [stop[:3]], 'to_stop_id': [stop[:3]], 'min_transfer_time': [0]})

            for i, row in possible_transfers.iterrows():
                # if it is interstation
                if row['from_stop_id'] == row['to_stop_id']:
                    if stop[3]=='N':
                        self.footpaths[stop].append((row['to_stop_id']+'S', row['min_transfer_time']))
                    if stop[3]=='S':
                        self.footpaths[stop].append((row['to_stop_id']+'N', row['min_transfer_time']))
                else:
                    self.footpaths[stop].append((row['to_stop_id']+'N', row['min_transfer_time']))
                    self.footpaths[stop].append((row['to_stop_id']+'S', row['min_transfer_time']))

    def find_minimum_times(self, from_station, from_time, K=4):
        #K = 4
        #p_s = ('201N', 10000)
        p_s = (from_station, from_time)

        # create multilabel
        tau_k = {stop: [np.inf] * (K+1) for stop in self.unique_stops}
        tau_k[p_s[0]][0] = p_s[1]

        Q = set([p_s[0]])

        for k in range(1,K+1):
            #print(f"Round {k}")

            # transfer from previous round
            for stop in self.unique_stops:
                tau_k[stop][k] = tau_k[stop][k-1]

            # Scan each route
            for route_id in self.routes.keys():
            #for route_id in routes_serving_Q:
                route_trips = self.route_trips_dict[route_id]    # all trips serving this route
                route_stops = self.routes[route_id]              # all stops on this trip

                earliest_trip = None
                hopped_on_station = None

                # Scan each stop in the route in order
                for stop in route_stops:
                    route_stops_idx = route_stops.index(stop)

                    # if we are on a trip, update stop with arrival time of the trip
                    if earliest_trip:
                        arrival_time = trip_stops[route_stops_idx][1]

                        # if we arrive sooner than previous estimate, update earliest arrival after k rounds
                        if arrival_time < tau_k[stop][k]:
                            tau_k[stop][k] = arrival_time
                            #Q.add(stop)

                        # if we arrived there earlier, update earliest_trip
                        if tau_k[stop][k-1] <= arrival_time:
                            earliest_trip = self.get_earliest_trip((stop, tau_k[stop][k-1]), route_id, route_trips)
                            trip_stops = route_trips[earliest_trip]
                            arrival_time = trip_stops[route_stops_idx][1]

                    # if we are not on a trip see if we can get on one
                    if not earliest_trip and tau_k[stop][k-1] < np.inf:
                        # find earliest trip
                        earliest_trip = self.get_earliest_trip((stop, tau_k[stop][k-1]), route_id, route_trips)
                        hopped_on_station = stop
                        # these are the (stop,time)'s of the earliest trip

                        if not earliest_trip:
                            continue

                        trip_stops = route_trips[earliest_trip]
                        #print(stop, tau_k[stop][k-1], route_id, route_trips)

            # compute footpaths
            for from_stop, transfer_list in self.footpaths.items():
                for to_stop, transfer_time in transfer_list:
                    if to_stop in self.unique_stops:
                        tau_k[to_stop][k] = min(tau_k[to_stop][k], tau_k[from_stop][k] + transfer_time)
                        #Q.add(to_stop)


        min_time = {}

        for stop, arrival_times in tau_k.items():
            min_time[stop] = {'time':min(arrival_times), 
                            'round':arrival_times.index(min(arrival_times))}
            
        return min_time