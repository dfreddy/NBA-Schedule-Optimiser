import csv
import string
import numpy as np
import pandas as pd
import statistics

# input is two team names
def distance_two_games(annealer, team_name_1, team_name_2):
    if team_name_1 == team_name_2:
        return 0

    team_1 = annealer.get_team_conf_div(team_name_1)
    team_2 = annealer.get_team_conf_div(team_name_2)
    if team_1[0] != team_2[0]:
        return 10
    if team_1[1] != team_2[1]:
        return 4
    else:
        return 2


# input is an entry from the teams_schedule dict
# output is total distance score for a given team
def calculate_team_distance_score(annealer, team_sch):
    i = 1
    score = 0
    while i < len(team_sch) - 1:
        score = score + distance_two_games(annealer, team_sch[i-1].get('home team'), team_sch[i].get('home team')) + distance_two_games(annealer, team_sch[i].get('home team'), team_sch[i+1].get('home team'))
        i = i + 1
    score = score + distance_two_games(annealer, team_sch[i-1].get('home team'), team_sch[i].get('home team'))
    return score


def handle_distance(annealer):
    ts = annealer.get_teams_schedules().copy()
    distance_scores_array = []
    score = 0

    for t in ts:
        team_score = calculate_team_distance_score(annealer, ts.get(t))
        annealer.add_distance_to_team_dict(t, team_score)
        distance_scores_array.append(team_score)
        score = score + team_score

    annealer.save_distance_array(distance_scores_array)
    return score, statistics.stdev(distance_scores_array)


def days_two_games(gameday_1, gameday_2):
    diff = abs(gameday_2 - gameday_1)

    if diff == 0:
        return 100000000000
    if diff == 1:
        return 8
    if diff == 2:
        return 5
    if diff == 3:
        return 2
    if diff == 4:
        return 1
    if diff >= 5:
        return 0


# input is an entry from the teams_schedule dict
# output is total distance score for a given team
def calculate_team_days_score(team_sch):
    i = 1
    score = 0

    score = score + days_two_games(team_sch[0].get('day'), team_sch[1].get('day'))

    while i < len(team_sch) - 1:
        score = score + days_two_games(team_sch[i - 1].get('day'), team_sch[i].get('day')) \
                + days_two_games(team_sch[i].get('day'), team_sch[i + 1].get('day'))
        i = i + 1
    score = score + days_two_games(team_sch[i - 1].get('day'), team_sch[i].get('day'))
    return score


# input -> teams_schedules
def handle_days(annealer):
    ts = annealer.get_teams_schedules().copy()
    days_scores_array = []
    score = 0

    for t in ts:
        team_score = calculate_team_days_score(ts.get(t))
        annealer.add_days_to_team_dict(t, team_score)
        days_scores_array.append(team_score)
        score = score + team_score

    annealer.save_days_array(days_scores_array)
    return score, statistics.stdev(days_scores_array)

def handle_neighbour(annealer, neighbour_ts, team_names, prev_score):
    teams = annealer.get_teams()
    ts = neighbour_ts

    days_scores_array = annealer.get_days_array().copy()
    # days_scores_array = []
    # for i in range(len(tmp_days_scores_array)):
    #    days_scores_array.append(tmp_days_scores_array[i])

    distance_scores_array = annealer.get_distance_array().copy()
    # distance_scores_array = []
    # for i in range(len(tmp_distance_scores_array)):
    #    distance_scores_array.append(tmp_distance_scores_array[i])

    prev_score = prev_score - statistics.stdev(days_scores_array) - statistics.stdev(distance_scores_array)

    for i in range(len(team_names)):
        team_dict = next(item for item in teams if item['team'] == team_names[i])
        prev_score = prev_score - team_dict['distance score'] - team_dict['days score']
        days_scores_array.remove(team_dict['days score'])
        distance_scores_array.remove(team_dict['distance score'])

    score = prev_score
    team_changes = []

    for t in ts:
        if t in team_names:
            # calculate days score
            team_days_score = calculate_team_days_score(ts.get(t))
            days_scores_array.append(team_days_score)
            score = score + team_days_score

            # calculate distance score
            team_distance_score = calculate_team_distance_score(annealer, ts.get(t))
            distance_scores_array.append(team_distance_score)
            score = score + team_distance_score

            team_changes.append([t, team_days_score, team_distance_score])

    score = score + statistics.stdev(days_scores_array) + statistics.stdev(distance_scores_array)

    return score, team_changes, days_scores_array, distance_scores_array

'''
days_score = handle_days(teams_schedules)
stdev_days = statistics.stdev(days_scores_array)
distance_score = handle_distance(teams_schedules)
stdev_distance = statistics.stdev(distance_scores_array)

print('days score\t\t-> ' + str(days_score) + '\tdays standard deviation\t\t= ' + str(stdev_days))
print('distance score\t-> ' + str(distance_score) + '\tdistance standard deviation\t= ' + str(stdev_distance))
print('schedule score\t-> ' + str(days_score + distance_score + stdev_days * len(teams) + stdev_distance * len(teams)))

print(dict(next(item for item in teams if item['team'] == 'New York Knicks')))
print(dict(next(item for item in teams if item['team'] == 'Los Angeles Lakers')))
'''


def evaluate(annealer):
    days_score, stdev_days_scores = handle_days(annealer)
    distance_score, stdev_distance_scores = handle_distance(annealer)

    return days_score + distance_score + stdev_days_scores + stdev_distance_scores


# teams_schedules, string of team names that changed, previous schedule's score
def evaluate_neighbour(annealer, neighbour_ts, team_names, prev_score):
    score, team_changes, days_scores_array, distance_scores_array = handle_neighbour(annealer, neighbour_ts, team_names, prev_score)

    return score, team_changes, days_scores_array, distance_scores_array
