import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os 

def prevResult(mainDF): 
    mainDF['Date'] = pd.to_datetime(mainDF['Date'])
    mainDF = mainDF.sort_values(by='Date', ascending=True)

    # Step 2: Build a unified team-games DataFrame from the home and away appearances.
    # Extract home team appearances:
    df_home = mainDF[['Date', 'HomeTeam', 'homePosition', 'homePoints', 'homePercentage', 'HomeTeamScore', 'AwayTeamScore', 'Year', 'Round']].copy()
    df_home = df_home.rename(columns={
        'HomeTeam': 'Team',
        'homePosition': 'position',
        'homePoints': 'points',
        'homePercentage': 'percentage',
        'HomeTeamScore': 'teamScore',
        'AwayTeamScore': 'opponentScore',
        'Year': 'year',
        'Round': 'round'
    })

    # Extract away team appearances:
    df_away = mainDF[['Date', 'AwayTeam', 'awayPosition', 'awayPoints', 'awayPercentage', 'HomeTeamScore', 'AwayTeamScore', 'Year', 'Round']].copy()
    df_away = df_away.rename(columns={
        'AwayTeam': 'Team',
        'awayPosition': 'position',
        'awayPoints': 'points',
        'awayPercentage': 'percentage',
        'AwayTeamScore': 'teamScore',
        'HomeTeamScore': 'opponentScore',
        'Year': 'year',
        'Round': 'round'
    })

    # Concatenate both so that every game for every team is in one DataFrame.
    df_team = pd.concat([df_home, df_away], ignore_index=True)

    # Sort by Team and Date so that each team’s games are in chronological order.
    df_team = df_team.sort_values(by=['Team', 'Date'], ascending=True)

    # Step 3: Create win_or_loss column based on the team score and opponent score.
    # If the team score is higher than the opponent score, it's a win (1), else it's a loss (0).
    df_team['win_or_loss'] = np.where(df_team['teamScore'] > df_team['opponentScore'], 1, 0)

    # Step 4: Shift the win_or_loss to get the previous game status.
    df_team['prev_win_or_loss'] = df_team.groupby('Team')['win_or_loss'].shift(1)

    # Step 5: Identify the first game of each year and set 'prev_win_or_loss' to NaN for those.
    # Reset win_or_loss for Round 1 of each year.
    mask_new_year_or_round1 = (df_team['round'] == 1)
    df_team.loc[mask_new_year_or_round1, 'prev_win_or_loss'] = np.nan

    # Step 6: Merge the win/loss information back into the original DataFrame for both Home and Away teams.
    # For home team win/loss info:
    home_prev = df_team[['Date', 'Team', 'prev_win_or_loss']].copy()
    home_prev = home_prev.rename(columns={
        'Team': 'HomeTeam',
        'prev_win_or_loss': 'previous_game_home_win_loss'
    })

    # Merge on Date and HomeTeam
    mainDF = pd.merge(mainDF, home_prev, on=['Date', 'HomeTeam'], how='left')

    # For away team win/loss info:
    away_prev = df_team[['Date', 'Team', 'prev_win_or_loss']].copy()
    away_prev = away_prev.rename(columns={
        'Team': 'AwayTeam',
        'prev_win_or_loss': 'previous_game_away_win_loss'
    })

    # Merge on Date and AwayTeam
    mainDF = pd.merge(mainDF, away_prev, on=['Date', 'AwayTeam'], how='left')
    return mainDF; 

def prevTable(mainDF):
    df_home = mainDF[['Date', 'HomeTeam', 'homePosition', 'homePoints', 'homePercentage']].copy()
    df_home = df_home.rename(columns={
        'HomeTeam': 'Team',
        'homePosition': 'position',
        'homePoints': 'points',
        'homePercentage': 'percentage'
    })

    # Extract away team appearances:
    df_away = mainDF[['Date', 'AwayTeam', 'awayPosition', 'awayPoints', 'awayPercentage']].copy()
    df_away = df_away.rename(columns={
        'AwayTeam': 'Team',
        'awayPosition': 'position',
        'awayPoints': 'points',
        'awayPercentage': 'percentage'
    })

    # Concatenate both so that every game for every team is in one DataFrame.
    df_team = pd.concat([df_home, df_away], ignore_index=True)

    # Sort by Team and Date so that each team’s games are in chronological order.
    df_team = df_team.sort_values(by=['Team', 'Date'], ascending=True)

    # Create a 'year' column for each game.
    df_team['year'] = df_team['Date'].dt.year

    # Step 3: For each team, shift the stats to create previous game columns.
    df_team['prev_position'] = df_team.groupby('Team')['position'].shift(1)
    df_team['prev_points'] = df_team.groupby('Team')['points'].shift(1)
    df_team['prev_percentage'] = df_team.groupby('Team')['percentage'].shift(1)

    # Also shift the year column to compare years.
    df_team['prev_year'] = df_team.groupby('Team')['year'].shift(1)

    # For games where the previous game is in a different year, set the previous game stats to NaN.
    mask_new_year = df_team['year'] != df_team['prev_year']
    df_team.loc[mask_new_year, ['prev_position', 'prev_points', 'prev_percentage']] = np.nan

    # (Optional) Drop the helper columns if you don't need them.
    df_team.drop(columns=['year', 'prev_year'], inplace=True)

    # Step 4: Merge the previous game info back into the original DataFrame.
    # For the home team info:
    home_prev = df_team[['Date', 'Team', 'prev_position', 'prev_points', 'prev_percentage']].copy()
    home_prev = home_prev.rename(columns={
        'Team': 'HomeTeam',
        'prev_position': 'previous_game_home_position',
        'prev_points': 'previous_game_home_points',
        'prev_percentage': 'previous_game_home_percentage'
    })

    # Merge on Date and HomeTeam
    mainDF = pd.merge(mainDF, home_prev, on=['Date', 'HomeTeam'], how='left')

    # For the away team info:
    away_prev = df_team[['Date', 'Team', 'prev_position', 'prev_points', 'prev_percentage']].copy()
    away_prev = away_prev.rename(columns={
        'Team': 'AwayTeam',
        'prev_position': 'previous_game_away_position',
        'prev_points': 'previous_game_away_points',
        'prev_percentage': 'previous_game_away_percentage'
    })

    # Merge on Date and AwayTeam
    mainDF = pd.merge(mainDF, away_prev, on=['Date', 'AwayTeam'], how='left')

    return mainDF

file_path = os.path.join('data', 'interim', 'fullSesData.csv')
mainDF = pd.read_csv(file_path)


mainDF['Date'] = pd.to_datetime(mainDF['Date'])
mainDF = mainDF.sort_values(by='Date', ascending=True)

mainDF = prevResult(mainDF)
mainDF = prevTable(mainDF)

# Removing stadiums which have low capacity or low usage
stadiums_to_remove = [
    'Bellerive Oval', 'Manuka Oval', 'Stadium Australia', 'Marrara Oval', 
    "Cazaly's Stadium", 'Eureka Stadium', 'Traeger Park', 'Wellington', 
    'Jiangwan Stadium', 'Norwood Oval', 'Blacktown', 'Riverway Stadium', 
    'Summit Sports Park', 'York Park'
]

mainDF = mainDF[~mainDF['Venue'].isin(stadiums_to_remove)]

# Define AFL rivalries
rivalries = [
    ('Collingwood', 'Essendon'),
    ('Collingwood', 'Carlton'),
    ('Adelaide', 'Port Adelaide'),
    ('Richmond', 'Essendon'),
    ('GWS', 'Sydney'),
    ('Richmond', 'Carlton'),
    ('Essendon', 'Carlton'),
    ('Geelong', 'Hawthorn'),
    ('Fremantle', 'West Coast'),
    ('Collingwood', 'Richmond')
]

# Create a new column "Rivalry" to label rivalry games
def add_rivalry_feature(mainDF, rivalries):
    # Initialize the "Rivalry" column to False
    mainDF['Rivalry'] = 0
    
    # Loop through each rivalry pair and label matches as part of the rivalry
    for team1, team2 in rivalries:
        # Find matches where either team1 or team2 is playing and mark them as rivalry
        mask = ((mainDF['HomeTeam'] == team1) & (mainDF['AwayTeam'] == team2)) | \
               ((mainDF['HomeTeam'] == team2) & (mainDF['AwayTeam'] == team1))
        mainDF.loc[mask, 'Rivalry'] = 1
    
    return mainDF

# Apply the rivalry feature to your mainDF
mainDF = add_rivalry_feature(mainDF, rivalries)
#Encoding
team_name_mapping = {
    'Adelaide': 1,
    'Brisbane Lions': 2,
    'Carlton': 3,
    'Collingwood': 4,
    'Essendon': 5,
    'Fremantle': 6,
    'Geelong': 7,
    'Gold Coast': 8,
    'Greater Western Sydney': 9,
    'Hawthorn': 10,
    'Melbourne': 11,
    'North Melbourne': 12,
    'Port Adelaide': 13,
    'Richmond': 14,
    'St Kilda': 15,
    'Sydney': 16,
    'West Coast': 17,
    'Western Bulldogs': 18
}

mainDF['HomeTeamEncode'] = mainDF['HomeTeam'].map(team_name_mapping)
mainDF['AwayTeamEncode'] = mainDF['AwayTeam'].map(team_name_mapping)

venue_name_mapping = {
    'M.C.G.': 1, 'Carrara': 2, 'Subiaco': 3, 'Docklands': 4, 
    'Football Park': 5, 'Gabba': 6, 'S.C.G.': 7, 'Kardinia Park': 8, 
    'Sydney Showground': 9, 'Adelaide Oval': 10, 
    'Perth Stadium': 11
}

mainDF['VenueEncode'] = mainDF['Venue'].map(venue_name_mapping)

#Introduce Stadium Capacity
stadium_capacity_mapping = {
    'M.C.G.': 100024,
    'Carrara': 25000,
    'Subiaco': 42922,
    'Docklands': 53359,
    'Football Park': 51240,
    'Gabba': 42000,
    'S.C.G.': 48000,
    'Kardinia Park': 40000,
    'Sydney Showground': 25000,
    'Adelaide Oval': 53583,
    'Perth Stadium': 60000
}

mainDF['StadiumCapacity'] = mainDF['Venue'].map(stadium_capacity_mapping)

#Create Day of the Week Variable
mainDF['DayC'] = mainDF['Date'].dt.day_name()

#Create Day Variable
mainDF['Day'] = mainDF['Date'].dt.dayofweek  # Extract the day of the week (Monday=0, Sunday=6)

#Create Month Variable
mainDF['Month'] = mainDF['Date'].dt.month

# Assuming StartTime is in 12-hour clock format (e.g., '7:20 PM')
mainDF['StartTime'] = pd.to_datetime(mainDF['StartTime'], format='%I:%M %p', errors='coerce')

# If StartTime is in 24-hour clock format (e.g., '19:20')
# mainDF['StartTime'] = pd.to_datetime(mainDF['StartTime'], format='%H:%M', errors='coerce')

# Extract minutes since midnight
mainDF['MinutesSinceMidnight'] = mainDF['StartTime'].dt.hour * 60 + mainDF['StartTime'].dt.minute

#Holiday/Seasonal Indicators

# Forward-fill NaN values within each HomeTeam group
mainDF['homePosition'] = mainDF.groupby('HomeTeam')['homePosition'].ffill()
mainDF['homePercentage'] = mainDF.groupby('HomeTeam')['homePercentage'].ffill()
mainDF['homePoints'] = mainDF.groupby('HomeTeam')['homePoints'].ffill()

mainDF['awayPosition'] = mainDF.groupby('AwayTeam')['awayPosition'].ffill()
mainDF['awayPercentage'] = mainDF.groupby('AwayTeam')['awayPercentage'].ffill()
mainDF['awayPoints'] = mainDF.groupby('AwayTeam')['awayPoints'].ffill()

mask_round1 = mainDF['Round'] == 1

# For first round games, set previous game stats to default values:
# For the Home Team:
mainDF.loc[mask_round1, 'previous_game_home_position'] = 9
mainDF.loc[mask_round1, 'previous_game_home_points'] = 0
mainDF.loc[mask_round1, 'previous_game_home_percentage'] = 0

# For the Away Team:
mainDF.loc[mask_round1, 'previous_game_away_position'] = 9
mainDF.loc[mask_round1, 'previous_game_away_points'] = 0
mainDF.loc[mask_round1, 'previous_game_away_percentage'] = 0

#Transform Betting Odds to Decimal Odds
mainDF['HomeProbability'] = 1 / mainDF['Home Odds']
mainDF['AwayProbability'] = 1 / mainDF['Away Odds']

#Special Round 
mainDF['isFinal'] = (mainDF['Round'] < 0).astype(int)


#Dealing with null values created by new features
mainDF['previous_game_home_win_loss'] = mainDF['previous_game_home_win_loss'].fillna(-1)
mainDF['previous_game_away_win_loss'] = mainDF['previous_game_away_win_loss'].fillna(-1)



output_file = os.path.join('data', 'processed', '12_23data.csv')
mainDF.to_csv(output_file, index=False)