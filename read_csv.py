import csv
import string
import numpy as np
import pandas as pd

thirty_days_months = [4, 6, 9, 11]
thirty_one_days_months = [1, 3, 5, 7, 8, 10, 0]  # month 0 corresponds to month 12 december


def set_conference_divisions(teams):
    out = []
    for t in teams:
        out_entry = {'error': 'team name missing'}
        if t == 'Sacramento Kings' or t == 'Golden State Warriors' or t == 'LA Clippers' or t == 'Los Angeles Lakers' or t == 'Phoenix Suns':
            out_entry = {
                'team': t,
                'conf': 'West',
                'div': 'Pacific'
            }
        if t == 'Portland Trail Blazers' or t == 'Utah Jazz' or t == 'Denver Nuggets' or t == 'Oklahoma City Thunder' or t == 'Minnesota Timberwolves':
            out_entry = {
                'team': t,
                'conf': 'West',
                'div': 'Northwest'
            }
        if t == 'San Antonio Spurs' or t == 'Dallas Mavericks' or t == 'Houston Rockets' or t == 'New Orleans Pelicans' or t == 'Memphis Grizzlies':
            out_entry = {
                'team': t,
                'conf': 'West',
                'div': 'Southwest'
            }
        if t == 'Miami Heat' or t == 'Orlando Magic' or t == 'Atlanta Hawks' or t == 'Charlotte Hornets' or t == 'Washington Wizards':
            out_entry = {
                'team': t,
                'conf': 'East',
                'div': 'Southeast'
            }
        if t == 'Milwaukee Bucks' or t == 'Chicago Bulls' or t == 'Indiana Pacers' or t == 'Detroit Pistons' or t == 'Cleveland Cavaliers':
            out_entry = {
                'team': t,
                'conf': 'East',
                'div': 'Central'
            }
        if t == 'Toronto Raptors' or t == 'Boston Celtics' or t == 'Brooklyn Nets' or t == 'Philadelphia 76ers' or t == 'New York Knicks':
            out_entry = {
                'team': t,
                'conf': 'East',
                'div': 'Atlantic'
            }
        out.append(out_entry)
    return out


def convert_dates(tdd):
    new_data_dates = []
    date_counter = 0
    prev_day = str(int(tdd[0].split("/")[0])-1)

    for i in range(len(tdd)):
        day = tdd[i].split("/")[0]
        month = tdd[i].split("/")[1]
        year = tdd[i].split("/")[2].split(" ")[0]

        # if the day of the game is the same as the previous one, use the same day counter
        if day == prev_day:
            new_data_dates.append(date_counter)
        # update day counter if the day changes
        else:
            # if the previous day was 30 and the previous month was a 30 day month
            if prev_day == '30' and day != '31' and int(month)-1 in thirty_days_months:
                date_counter = date_counter + int(day)
                new_data_dates.append(date_counter)
                prev_day = day
                continue

            # if the previous day was 31 and the previous month was a 31 day month
            if prev_day == '31' and int(month)-1 in thirty_one_days_months:
                date_counter = date_counter + int(day)
                new_data_dates.append(date_counter)
                prev_day = day
                continue

            # if the previous day was 29th february on a leap year
            if prev_day == '29' and int(month)-1 == 2 and int(year) % 4 == 0 and int(year) % 100 == 0 and int(year) % 400 == 0:
                date_counter = date_counter + int(day)
                new_data_dates.append(date_counter)
                prev_day = day
                continue

            # if the previous day was 28th february on a non leap year
            if prev_day == '28' and day != '29' and int(month)-1 == 2:
                date_counter = date_counter + int(day)
                new_data_dates.append(date_counter)
                prev_day = day
                continue

            # if it's not the end of the month, update day counter normally
            date_counter = date_counter + int(day)-int(prev_day)
            new_data_dates.append(date_counter)
            prev_day = day

    return new_data_dates


data = pd.read_csv('nba2018_schedule.csv')

tmp_data_dates = data['Date'].values
data_days = convert_dates(tmp_data_dates)
data_home_teams = data['Home Team'].values
data_away_teams = data['Away Team'].values
data_array = []

data_teams = set(data_home_teams)

for i in range(len(data_days)):
    data_array_entry = {
        'day': data_days[i],
        'home team': data_home_teams[i],
        'away team': data_away_teams[i]
    }
    data_array.append(data_array_entry)


# saving data in csv file
with open('data.csv', 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['day', 'home team', 'away team'])
    for i in range(len(data_array)):
        csv_writer.writerow([data_array[i].get('day'),
                             data_array[i].get('home team'), data_array[i].get('away team')])

teams_with_confs = set_conference_divisions(data_teams)

with open('teams.csv', 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['team', 'conference', 'division'])
    for i in range(len(teams_with_confs)):
        csv_writer.writerow([teams_with_confs[i].get('team'), teams_with_confs[i].get('conf'), teams_with_confs[i].get('div')])


