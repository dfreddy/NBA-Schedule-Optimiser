import csv
import string
import numpy as np
import pandas as pd
import statistics
from random import *
import math
import time
from operator import itemgetter
import matplotlib.pyplot as plt
import copy
from evaluator import evaluate, evaluate_neighbour

class SimAnneal:

    # contains an entry for each team, with team name, conference, division, (after evaluation) distance and days score
    teams = []
    # a dict with team names as keys, each containing the team's schedules
    teams_schedules = {}

    days_scores_array = []
    distance_scores_array = []

    score = 0
    initial_temperature = 250
    temperature = 0
    temperature_cycle = 200
    max_cycles = 10000

    best_schedule = None
    best_score = 0
    best_days_array = None
    best_distance_array = None
    best_teams_dict = None
    reset_cycle = 400

    def __init__(self):
        data = pd.read_csv('data.csv')  # array containing the whole schedule
        data_days = data['day'].values
        data_home_teams = data['home team'].values
        data_away_teams = data['away team'].values

        f = csv.DictReader(open("teams.csv"))
        for row in f:
            self.teams.append(row)

        # create individual schedule arrays for each team
        for t in self.teams:
            team_schedule = []
            for k in range(len(data_days)):  # when found the team in the main schedule, add that schedule entry to the team schedule
                if t.get('team') == data_home_teams[k] or t.get('team') == data_away_teams[k]:
                    team_schedule_entry = {
                        'day': data_days[k],
                        'home team': data_home_teams[k],
                        'away team': data_away_teams[k]
                    }
                    team_schedule.append(team_schedule_entry)
            self.teams_schedules[t.get('team')] = team_schedule  # add that team's schedule to the dict of team schedules

    def get_teams_schedules(self):
        return copy.deepcopy(self.teams_schedules)

    def get_best_schedule(self):
        return self.best_schedule

    def save_teams_schedules(self, neighbour_ts):
        self.teams_schedules = neighbour_ts

    def get_teams(self):
        return self.teams

    def add_distance_to_team_dict(self, team, distance):
        team_dict = next(item for item in self.teams if item['team'] == team)
        team_dict['distance score'] = distance

    def add_days_to_team_dict(self, team, days):
        team_dict = next(item for item in self.teams if item['team'] == team)
        team_dict['days score'] = days

    def save_best_team_dict(self):
        self.best_teams_dict = []
        for item in self.teams:
            entry = {'team': item['team'],
                     'conference': item['conference'],
                     'division': item['division'],
                     'distance score': item['distance score'],
                     'days score': item['days score']}
            self.best_teams_dict.append(entry)
        return True

    def load_best_team_dict(self):
        self.teams = []
        for item in self.best_teams_dict:
            entry = {'team': item['team'],
                     'conference': item['conference'],
                     'division': item['division'],
                     'distance score': item['distance score'],
                     'days score': item['days score']}
            self.teams.append(entry)
        return True

    def get_team_conf_div(self, team):
        team_dict = next(item for item in self.teams if item['team'] == team)
        return [team_dict.get('conference'), team_dict.get('division')]

    def save_days_array(self, array):
        self.days_scores_array = array

    def save_distance_array(self, array):
        self.distance_scores_array = array

    def save_score(self, n_score):
        self.score = n_score

    def get_days_array(self):
        return self.days_scores_array

    def get_distance_array(self):
        return self.distance_scores_array

    def update_temperature(self, k):
        # self.temperature = self.temperature / math.log(k+1)
        self.temperature = 0.8 * self.temperature

    def reset_temperature(self):
        # self.initial_temperature = 0.8 * self.initial_temperature
        self.temperature = self.initial_temperature

    def acceptance_prob(self, cost):
        # cost must b positive
        prob = math.exp(- cost / self.temperature)
        print('Acceptance prob: ' + str(round(prob, 3)))
        return prob

    def anneal(self):
        # get first evaluation
        # self.tmp_teams_schedule = copy.deepcopy(self.teams_schedules)
        self.score = evaluate(self)
        self.best_score = self.score
        self.best_schedule = self.teams_schedules
        initial_score = self.score

        self.reset_temperature()

        history = [initial_score]

        # loop for max_cycles times
        k = 1
        n = 1
        b = 1
        while n <= self.max_cycles:
            print('Cycle number ' + str(n))

            # find a neighbour wit one of the three methods
            neighbour_rand = randint(1, 10)
            # neighbour_rand = 10
            neighbour = None
            teams_affected = []
            if neighbour_rand <= 3:
                neighbour, teams_affected = neighbour_date(self)
            if 3 < neighbour_rand < 10:
                neighbour, teams_affected = neighbour_home_away(self)
            if neighbour_rand == 10:
                neighbour, teams_affected = neighbour_switch_games(self)

            # evaluate neighbour
            n_score, team_changes, days_scores_array, distance_scores_array = evaluate_neighbour(self, neighbour, teams_affected, self.score)

            print('\nNeighbour score: ' + str(n_score))

            # measure difference with temperature
            cost = n_score - self.score
            print('\nCurrent temperature: ' + str(round(self.temperature, 3)))
            print('Neighbour cost: ' + str(round(cost, 3)))
            accepted = False
            if cost <= 0:
                accepted = True
            else:
                r = random()
                print('Random odd: ' + str(round(r, 3)))
                if r <= self.acceptance_prob(cost):
                    accepted = True
            # update annealer attributes if accepted
            if accepted:
                print('\nYES - Accepted Neighbour State\n----------------------------------------\n')

                k = 1

                self.save_teams_schedules(neighbour)
                self.save_days_array(days_scores_array)
                self.save_distance_array(distance_scores_array)
                self.save_score(n_score)

                for i in range(len(team_changes)):
                    self.add_days_to_team_dict(team_changes[i][0], team_changes[i][1])
                    self.add_distance_to_team_dict(team_changes[i][0], team_changes[i][2])

                if n_score < self.best_score:
                    self.best_score = n_score
                    self.best_schedule = neighbour
                    self.best_days_array = self.days_scores_array
                    self.best_distance_array = self.distance_scores_array
                    done = False
                    done = self.save_best_team_dict()
                    while not done:
                        0

            else:
                print("\nNO - Didn't Accept Neighbour State\n----------------------------------------\n")
                k = k + 1

            history.append(self.score)

            # update temperature, reset temperature if not found a better solution after x cycles
            if k == self.temperature_cycle:
                k = 1
                self.reset_temperature()
            else:
                self.update_temperature(k)

            # reset to best schedule after x cycles
            b = b + 1
            if b > self.reset_cycle:
                self.score = self.best_score
                self.teams_schedules = self.best_schedule
                self.days_scores_array = self.best_days_array
                self.distance_scores_array = self.best_distance_array
                done = False
                done = self.load_best_team_dict()
                while not done:
                    0
                b = 1

            n = n + 1

        print('Initial score: ' + str(initial_score))
        print('Final score: ' + str(self.score))
        print('Schedule improvement: ' + str(round((initial_score - self.score) * 100 / initial_score, 3)) + '%')
        print('\nBest score: ' + str(self.best_score))
        print('Best schedule improvement: ' + str(round((initial_score - self.best_score) * 100 / initial_score, 3)) + '%')

        plt.plot(history)
        plt.ylabel('Score')
        plt.xlabel('Iteration')
        plt.show()


# changes date on a team's game by up to 2 days
# returns the neighbour state and an array of the effected teams
def neighbour_date(annealer):
    neighbour_ts = copy_schedule(annealer)
    # neighbour_ts = dict(annealer.teams_schedules)

    # pick random team, game and date change
    date_condition = True
    while date_condition:
        team_index = randint(0, len(annealer.teams)-1)
        team_name = annealer.teams[team_index]['team']
        game_index = randint(0, len(neighbour_ts.get(team_name))-1)

        for t in neighbour_ts:
            if t != team_name:
                games_list = neighbour_ts.get(t)
                if neighbour_ts.get(team_name)[game_index] in games_list:
                    date_change = choice([i for i in range(-20, 20) if i not in [0]])
                    team_name_2 = t
                    game_index_2 = games_list.index(neighbour_ts.get(team_name)[game_index])

                    # make sure games don't overlap
                    if not has_game_in_date(neighbour_ts.get(team_name), neighbour_ts.get(team_name)[game_index]['day'] + date_change) and \
                            not has_game_in_date(neighbour_ts.get(team_name_2), neighbour_ts.get(team_name_2)[game_index_2]['day'] + date_change) and \
                            0 < neighbour_ts.get(team_name_2)[game_index_2]['day'] + date_change < 178:

                        print('Changing date by ' + str(date_change) + ' days for ' + str(neighbour_ts.get(team_name)[game_index]))
                        games_list[game_index_2]['day'] = games_list[game_index_2]['day'] + date_change
                        neighbour_ts.get(team_name)[game_index]['day'] = games_list[game_index_2]['day']

                        # update day order of the team schedule
                        neighbour_ts.get(team_name).sort(key=itemgetter('day'))
                        neighbour_ts.get(team_name_2).sort(key=itemgetter('day'))

                        print(team_name + ' -> ' + str(neighbour_ts.get(team_name)[game_index]))
                        print(team_name_2 + ' -> ' + str(neighbour_ts.get(team_name_2)[game_index_2]))

                        date_condition = False
                        break

    return neighbour_ts, [team_name, team_name_2]

# switches home/away on a team's game
# returns the neighbour state and an array of the effected teams
def neighbour_home_away(annealer):
    neighbour_ts = copy_schedule(annealer)
    # neighbour_ts = dict(annealer.teams_schedules)

    # pick random team and its game
    team_index = randint(0, len(annealer.teams) - 1)
    team_name = annealer.teams[team_index]['team']
    game_index = randint(0, len(neighbour_ts.get(team_name)) - 1)
    team_name_2 = None

    print('Switching home away teams for game ' + str(neighbour_ts.get(team_name)[game_index]))

    for t in neighbour_ts:
        if t != team_name:
            games_list = neighbour_ts.get(t)
            if neighbour_ts.get(team_name)[game_index] in games_list:
                team_name_2 = t
                game_index_2 = games_list.index(neighbour_ts.get(team_name)[game_index])

                new_game = {'day': games_list[game_index_2].get('day'),
                            'home team': games_list[game_index_2].get('away team'),
                            'away team': games_list[game_index_2].get('home team')}

                games_list[game_index_2] = new_game
                neighbour_ts.get(team_name)[game_index] = new_game

                if games_list[game_index_2]['home team'] == games_list[game_index_2]['away team']:
                    print('ERROR: ' + str(neighbour_ts.get(team_name)[game_index]))
                    return None

                # update day order of the team schedule
                neighbour_ts.get(team_name).sort(key=itemgetter('day'))
                neighbour_ts.get(team_name_2).sort(key=itemgetter('day'))

                print(team_name + ' -> ' + str(neighbour_ts.get(team_name)[game_index]))
                print(team_name_2 + ' -> ' + str(neighbour_ts.get(team_name_2)[game_index_2]))

                break
    new_game_index = game_index
    found = False
    while new_game_index == game_index or found is False:
        new_game_index = randint(0, len(neighbour_ts.get(team_name)) - 1)
        if neighbour_ts.get(team_name)[new_game_index].get('away team') == team_name_2 or neighbour_ts.get(team_name)[new_game_index].get('home team') == team_name_2:
            found = True

    print('Switching home away teams for game ' + str(neighbour_ts.get(team_name)[new_game_index]))

    for t in neighbour_ts:
        if t != team_name:
            games_list = neighbour_ts.get(t)
            if neighbour_ts.get(team_name)[new_game_index] in games_list:
                team_name_2 = t
                game_index_2 = games_list.index(neighbour_ts.get(team_name)[new_game_index])

                new_game = {'day': games_list[game_index_2].get('day'),
                            'home team': games_list[game_index_2].get('away team'),
                            'away team': games_list[game_index_2].get('home team')}

                games_list[game_index_2] = new_game
                neighbour_ts.get(team_name)[new_game_index] = new_game

                if games_list[game_index_2]['home team'] == games_list[game_index_2]['away team']:
                    print('ERROR: ' + str(neighbour_ts.get(team_name)[new_game_index]))
                    return None

                # update day order of the team schedule
                neighbour_ts.get(team_name).sort(key=itemgetter('day'))
                neighbour_ts.get(team_name_2).sort(key=itemgetter('day'))

                print(team_name + ' -> ' + str(neighbour_ts.get(team_name)[new_game_index]))
                print(team_name_2 + ' -> ' + str(neighbour_ts.get(team_name_2)[game_index_2]))

                break

    return neighbour_ts, [team_name, team_name_2]

# switches dates between a team's two games
# returns the neighbour state and an array of the effected teams
def neighbour_switch_games(annealer):
    neighbour_ts = copy_schedule(annealer)
    # neighbour_ts = dict(annealer.teams_schedules)

    # pick random team and the games to switch dates
    overlap_1 = True
    overlap_2 = True

    team_name = ''
    team_name_2 = ''
    team_name_3 = ''

    # will only update later if theres no overlaps
    game = None
    game_2 = None
    switch_game = None
    switch_game_2 = None

    while overlap_1 or overlap_2:

        team_index = randint(0, len(annealer.teams) - 1)
        team_name = annealer.teams[team_index]['team']

        game_index = randint(0, len(neighbour_ts.get(team_name)) - 1)
        switch_game_index = 0
        while True:
            switch_game_index = randint(0, len(neighbour_ts.get(team_name)) - 1)
            if switch_game_index != game_index:
                break

        date = neighbour_ts.get(team_name)[game_index]['day']
        switch_date = neighbour_ts.get(team_name)[switch_game_index]['day']

        game = neighbour_ts.get(team_name)[game_index]
        switch_game = neighbour_ts.get(team_name)[switch_game_index]

        for t in neighbour_ts:
            if t != team_name:
                games_list = neighbour_ts.get(t)
                if neighbour_ts.get(team_name)[game_index] in games_list:
                    team_name_2 = t
                    game_index_2 = games_list.index(neighbour_ts.get(team_name)[game_index])

                    overlap_1 = has_game_in_date(games_list, switch_date)

                    if overlap_1:
                        continue

                    # get game to update if no overlap
                    game_2 = games_list[game_index_2]

                    break

        for t in neighbour_ts:
            if t != team_name:
                games_list = neighbour_ts.get(t)
                if neighbour_ts.get(team_name)[switch_game_index] in games_list:
                    team_name_3 = t
                    game_index_3 = games_list.index(neighbour_ts.get(team_name)[switch_game_index])

                    overlap_2 = has_game_in_date(games_list, date)

                    if overlap_2:
                        continue

                    # get other game to update if theres no overlap
                    switch_game_2 = games_list[game_index_3]

                    break

    print('\nSwitching games for ' + team_name)
    print(game)
    print(switch_game)

    game['day'] = switch_date
    game_2['day'] = switch_date
    switch_game['day'] = date
    switch_game_2['day'] = date

    print(team_name + ' -> ' + str(neighbour_ts.get(team_name)[game_index]))
    print(team_name + ' -> ' + str(neighbour_ts.get(team_name)[switch_game_index]))

    # update day order of the team schedule
    neighbour_ts.get(team_name).sort(key=itemgetter('day'))
    neighbour_ts.get(team_name_2).sort(key=itemgetter('day'))
    neighbour_ts.get(team_name_3).sort(key=itemgetter('day'))

    # for row in neighbour_ts.get(team_name):
    #     print(row)

    tc = [team_name, team_name_2, team_name_3]
    if team_name_2 == team_name_3:
        tc = [team_name, team_name_2]

    return neighbour_ts, tc

def copy_schedule(annealer):
    tmp_ts = annealer.teams_schedules
    neighbour_ts = {}
    for row in tmp_ts:
        team_schedule = []
        for k in range(len(tmp_ts.get(row))):
                team_schedule_entry = {
                    'day': tmp_ts.get(row)[k]['day'],
                    'home team': tmp_ts.get(row)[k]['home team'],
                    'away team': tmp_ts.get(row)[k]['away team']
                }
                team_schedule.append(team_schedule_entry)
        team_schedule.sort(key=itemgetter('day'))
        neighbour_ts[row] = team_schedule

    return neighbour_ts

def has_game_in_date(games_list, date):
    for i in range(len(games_list)):
        if games_list[i]['day'] == date:
            # print('OVERLAP')
            return True
    return False

def export_schedule(annealer):
    print('\nExporting schedule to csv...')
    ts = annealer.get_best_schedule()
    schedule = []

    for row in ts:
        for i in range(len(ts.get(row))):
            if ts.get(row)[i] not in schedule:
                schedule.append(ts.get(row)[i])

    schedule.sort(key=itemgetter('day'))

    with open('result.csv', 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['day', 'home team', 'away team'])
        for i in range(len(schedule)):
            csv_writer.writerow(
                [schedule[i].get('day'), schedule[i].get('home team'), schedule[i].get('away team')])


annealer = SimAnneal()


start_time = time.time()
annealer.anneal()
elapsed_time = time.time() - start_time
print('\nElapsed time: ' + str(round(elapsed_time, 5)) + ' seconds')

print('\nEval score: ' + str(evaluate(annealer)))

export_schedule(annealer)

'''
annealer.score = evaluate(annealer)
print("\nSchedule score = " + str(annealer.score))

print('\nGenerating neighbour by changing game date...\n')
nts, t_array = neighbour_date(annealer)
# nts, t_array = neighbour_home_away(annealer)
# nts, t_array = neighbour_switch_games(annealer)

n_score, team_changes, days_scores_array, distance_scores_array = evaluate_neighbour(annealer, nts, t_array, annealer.score)

diff = n_score - annealer.score

print("\nNeighbour score = " + str(n_score))
print("Difference = " + str(diff))

# if diff > 0:
print('\nAccepted Neighbour State\n----------------------------------------')

annealer.save_days_array(days_scores_array)
annealer.save_distance_array(distance_scores_array)

for i in range(len(team_changes)):
    annealer.add_days_to_team_dict(team_changes[i][0], team_changes[i][1])
    annealer.add_distance_to_team_dict(team_changes[i][0], team_changes[i][2])

# ANOTHER ONE
print('\nGenerating neighbour by switching home/away teams...\n')
# nts, t_array = neighbour_date(annealer)
nts, t_array = neighbour_home_away(annealer)
# nts, t_array = neighbour_switch_games(annealer)

n_score, team_changes, days_scores_array, distance_scores_array = evaluate_neighbour(annealer, nts, t_array, annealer.score)

diff = n_score - annealer.score

print("\nNeighbour score = " + str(n_score))
print("Difference = " + str(diff))

# if diff > 0:
print('\nAccepted Neighbour State\n----------------------------------------')

annealer.save_days_array(days_scores_array)
annealer.save_distance_array(distance_scores_array)

for i in range(len(team_changes)):
    annealer.add_days_to_team_dict(team_changes[i][0], team_changes[i][1])
    annealer.add_distance_to_team_dict(team_changes[i][0], team_changes[i][2])

# ANOTHER ONE
print('\nGenerating neighbour by switching games...\n')
# nts, t_array = neighbour_date(annealer)
# nts, t_array = neighbour_home_away(annealer)
nts, t_array = neighbour_switch_games(annealer)

n_score, team_changes, days_scores_array, distance_scores_array = evaluate_neighbour(annealer, nts, t_array, annealer.score)

diff = n_score - annealer.score

print("\nNeighbour score = " + str(n_score))
print("Difference = " + str(diff))

# if diff > 0:
print('\nAccepted Neighbour State\n----------------------------------------')

annealer.save_days_array(days_scores_array)
annealer.save_distance_array(distance_scores_array)

for i in range(len(team_changes)):
    annealer.add_days_to_team_dict(team_changes[i][0], team_changes[i][1])
    annealer.add_distance_to_team_dict(team_changes[i][0], team_changes[i][2])
'''
