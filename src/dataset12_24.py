import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np


def scrape_afl_ladder(yearStart, yearEnd):
    #Initialise the dataframe 
    df = pd.DataFrame(columns = ["Position", "Team", "Played", "Points", "Percentage", "Round", "Year"])

    # Loop through each year
    for year in range(yearStart, yearEnd + 1):
        url = f"https://afltables.com/afl/seas/{year}.html"

        # Send a GET request to the URL
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")

        table = soup.find_all("td", {"width": "15%", "valign": "top"})

        column_data = table[0].find_all("tr")

        #Opening Round
        round = 1
        teams_in_round = set()
        for row_index, row in enumerate(column_data[1:], start=1):
                row_data = row.find_all("td")
                length = len(df)
                individual_row = [row_index] + [data.text.strip() for data in row_data] + [1] + [year]
                #print(individual_row)
                df.loc[length] = individual_row
                teams_in_round.add(individual_row[1])

        all_teams = {"GW", "SY", "GC", "CA", "PA", "FR", "ES", "ME", "GE", "AD", "SK", "BL", "HW", "RI", "CW", "NM", "WB", "WC"}  # Replace with all possible team names
        missing_teams = all_teams - teams_in_round

        for team in missing_teams:
            # Append a row for the missing team with all values set to 0
            individual_row = [len(df) + 1, team, 0, 0, 0, round, 2024]  # Adjust based on your table structure
            df.loc[len(df)] = individual_row
        
        #Loop through the remaining rounds
        for round in range(1, len(table)):
            column_data = table[round].find_all("tr")
            for row_index, row in enumerate(column_data[1:], start=1):
                row_data = row.find_all("td")
                length = len(df)
                individual_row = [row_index] + [data.text.strip() for data in row_data] + [round + 1] + [year]
                df.loc[length] = individual_row

    return df
    
def mergeDataset(mainDF, df):
    #Merge Home Team Data
    mainDF = mainDF.merge(
        df[['Year', 'Round', 'Team', 'Position', 'Points', 'Percentage']],
        left_on=['Year', 'Round', 'HomeTeam'],  # Match Year, Round, and Home Team
        right_on=['Year', 'Round', 'Team'],
        how='left'
    ).rename(columns={'Position': 'homePosition', 'Points': 'homePoints', 'Percentage': 'homePercentage'}) \
    .drop(columns=['Team'])

    #Merge Away Team Data
    mainDF = mainDF.merge(
        df[['Year', 'Round', 'Team', 'Position', 'Points', 'Percentage']],
        left_on=['Year', 'Round', 'AwayTeam'],  # Match Year, Round, and Home Team
        right_on=['Year', 'Round', 'Team'],
        how='left'
    ).rename(columns={'Position': 'awayPosition', 'Points':'awayPoints','Percentage': 'awayPercentage'}) \
    .drop(columns=['Team'])

    return mainDF

def mergeBetDataset(mainDF, bettingDf):
    # Strip whitespace from team names in bettingDf
    bettingDf['HomeTeam'] = bettingDf['HomeTeam'].str.strip()
    bettingDf['AwayTeam'] = bettingDf['AwayTeam'].str.strip()

    # Merge Betting Data on 'HomeTeamScore', 'AwayTeamScore', 'HomeTeam', 'AwayTeam'
    mainDF = mainDF.merge(
        bettingDf[['Home Score', 'Away Score', 'HomeTeam', 'AwayTeam', 'Home Odds', 'Away Odds']],  # Include necessary columns
        left_on=['HomeTeamScore', 'AwayTeamScore', 'HomeTeam', 'AwayTeam'],  # Match on 'HomeTeamScore', 'AwayTeamScore', 'HomeTeam', 'AwayTeam'
        right_on=['Home Score', 'Away Score', 'HomeTeam', 'AwayTeam'],  # Match on 'Home Score', 'Away Score', 'HomeTeam', 'AwayTeam'
        how='left'  # Left join to keep all rows from mainDF
    ).drop(columns=['Home Score', 'Away Score'])  # Drop duplicate columns

    return mainDF




def clean_round(value):
    try:
        return int(value)  # Try converting to int
    except ValueError:
        if value == "GF":
            return -1
        if value == "PF":
            return -2
        if value == "SF":
            return -3
        if value == "QF":
            return -4
        if value == "EF":
            return -5


#For 2012 - 2024
df = scrape_afl_ladder(2012, 2024)
mainDF = pd.read_csv(r'C:\Users\raadr\OneDrive\Desktop\AflAnalysis-\data\external\games.csv')

#Setting both data frames to uniform names
team_name_mapping = {
    'AD' : 'Adelaide',
    'BL' : 'Brisbane Lions',
    'CA' : 'Carlton',
    'CW' : 'Collingwood',
    'ES' : 'Essendon',
    'FR' : 'Fremantle',
    'GE' : 'Geelong',
    'GC' : 'Gold Coast',
    'GW' : 'Greater Western Sydney',
    'HW' : 'Hawthorn',
    'ME' : 'Melbourne',
    'NM' : 'North Melbourne',
    'PA' : 'Port Adelaide',
    'RI' : 'Richmond',
    'SK' : 'St Kilda',
    'SY' : 'Sydney',
    'WC' : 'West Coast',
    'WB' : 'Western Bulldogs'
}
df['Team'] = df['Team'].replace(team_name_mapping)

#Remove R from Round column
mainDF['Round'] = mainDF['Round'].str.replace('R', '')
mainDF['Round'] = mainDF['Round'].apply(clean_round)
mainDF['Round'] = mainDF['Round'].astype(int) 
#Merging Data
mainDF = mergeDataset(mainDF, df)

mainDF.to_csv(r'C:\Users\raadr\OneDrive\Desktop\AflAnalysis-\data\raw\rawData12_24.csv', index=False)

team_name_bet_mapping = {
    'Brisbane' : 'Brisbane Lions',
    'GWS Giants' : 'Greater Western Sydney'
}
bettingDf = pd.read_csv(r'C:\Users\raadr\OneDrive\Desktop\AflAnalysis-\data\external\aflOdds.csv')
bettingDf['HomeTeam'] = bettingDf['HomeTeam'].replace(team_name_bet_mapping)
bettingDf['AwayTeam'] = bettingDf['AwayTeam'].replace(team_name_bet_mapping)

mainDF = mergeBetDataset(mainDF,bettingDf)

mainDF.to_csv(r'C:\Users\raadr\OneDrive\Desktop\AflAnalysis-\data\raw\rawData12_24Complete.csv', index=False)