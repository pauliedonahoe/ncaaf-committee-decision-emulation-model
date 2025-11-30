#!/usr/bin/env python
# coding: utf-8

# In[19]:


# Description - The below code does the following:
### 1 - inputs a list of NCAAF teams with their TeamRankings predictive ratings, 
### 2 - generates a randomized schedule with random pairings of teams, separated into different "weeks" of the season
### 3 - simulates the full season using the randomized schedule, running a random normal distribution for each game
### 4 - uses the method of iteratively calculated network-based rating to calculate final ratings for each team and outputs, 
###     in an attempt to emulate the human decision-making from the NCAAF ranking committee
### 5 - provides an additional set of optional outputs to see further details of the simulated season, 
###     and/or generate input for NCAAF Tournament Simulator script in Python

## Import packages
import pandas as pd
import numpy as np
import random

## Inputs
teams_list_file_path = '/YOUR_FILE_PATH/cfb_teams_list.csv'

## Main Output
final_team_rankings_output_excel_path = '/YOUR_FILE_PATH/final_team_ratings.xlsx'

## Optional Outputs
schedule_excel_file_path = '/YOUR_FILE_PATH/schedule_output.xlsx'
matrix_excel_file_path = '/YOUR_FILE_PATH/matrix_output.xlsx'
opponent_details_excel_path = '/YOUR_FILE_PATH/team_opponent_ratings.xlsx'
tournament_data_csv_path = '/YOUR_FILE_PATH/tournament_data.csv'


# In[2]:


# Read the input CSV file into a pandas DataFrame

df = pd.read_csv(teams_list_file_path)

print(df)


# In[3]:


# Define function to generate random schedules by creating random pairings of teams

def generate_season_pairings(df, weeks=7, random_seed=None):
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    teams = df["Team"].tolist()
    
    # Sanity check: must have an even number of teams
    if len(teams) % 2 != 0:
        raise ValueError("Number of teams must be even to form pairs.")
    
    schedule = {}

    for week in range(1, weeks + 1):
        shuffled = teams.copy()
        random.shuffle(shuffled)
        
        pairings = [(shuffled[i], shuffled[i + 1]) for i in range(0, len(shuffled), 2)]
        schedule[f"Week {week}"] = pairings

    return schedule


# In[4]:


# Generate randomized schedule for your preferred number of weeks
schedule = generate_season_pairings(df, weeks=7, random_seed=42)


# In[5]:


# Schedule check - view entire schedule in the console

for week, games in schedule.items():
    print(f"{week}:")
    for match in games:
        print(f"  {match[0]} vs {match[1]}")
    print()


# In[8]:


# Simulates full season schedule, using random normal distributions for each game

# ---- STEP 1: Convert schedule to a DataFrame ----
schedule_df = pd.DataFrame([
    {"Week": week, "Away": away, "Home": home}
    for week, games in schedule.items()
    for away, home in games
])

# ---- STEP 2: Merge ratings for both Away and Home teams ----
schedule_df = schedule_df.merge(df.rename(columns={"Team": "Away", "Rating": "Away_Rating"}), on="Away")
schedule_df = schedule_df.merge(df.rename(columns={"Team": "Home", "Rating": "Home_Rating"}), on="Home")

# ---- STEP 3: Calculate Spread = (Home_Rating - Away_Rating) + 2.6 ----
schedule_df["Spread"] = (schedule_df["Home_Rating"] - schedule_df["Away_Rating"]) + 2.6

# ---- STEP 4: Generate random Score_Differential for each matchup ----
score_diffs = []
for _, row in schedule_df.iterrows():
    while True: #while (random_value<=0.5 or random_value>=0.5):
        random_value = np.random.normal(loc=row["Spread"], scale=13.89)
        if -0.5 <= random_value <= 0.5:
            continue
        break
    score_diffs.append(random_value)

schedule_df["Score_Differential"] = score_diffs

# ---- Optional preview ----
print(schedule_df.head(10))


# In[10]:


# For simulated season results, create matrix dataframe (optionally can generate as output at bottom) to help with final iterative calculations

teams = sorted(set(schedule_df['Home'].unique()) | set(schedule_df['Away'].unique()))

matrix_df = pd.DataFrame(index=teams, columns=teams)

for team_a in teams:
    for team_b in teams:
        # Filter the dataframe for the matching combinations of team_a and team_b
        subset_schedule_df = schedule_df[((schedule_df['Home'] == team_a) & (schedule_df['Away'] == team_b)) | ((schedule_df['Home'] == team_b) & (schedule_df['Away'] == team_a))]

        # Calculate the value for each row in the schedule data
        values = []
        for _, row in subset_schedule_df.iterrows():
            value = 0.875 if (row['Home'] == team_a) and (row['Score_Differential'] > 0) else \
                    -1.125 if (row['Home'] == team_a) and (row['Score_Differential'] < 0) else \
                    -0.875 if (row['Home'] == team_b) and (row['Score_Differential'] > 0) else \
                    1.125 if (row['Home'] == team_b) and (row['Score_Differential'] < 0) else 0

            values.append(value)

        # Sum the values for the matching combinations
        total_value = sum(values)

        # Update the matrix with the total value
        matrix_df.at[team_a, team_b] = total_value
        matrix_df.at[team_b, team_a] = -total_value

# Fill NaN values with 0
matrix_df = matrix_df.fillna(0)

# ---- NEW SECTION: Add row-sum column ----
matrix_df['Adjusted Win Total'] = matrix_df.sum(axis=1)

# ---- Optional preview ----
print(matrix_df.head(10))


# In[11]:


# Final team rating calculation, using iterative rating updating methodology

# Strip whitespace from column names
schedule_df.columns = schedule_df.columns.str.strip()
schedule_df['Home'] = schedule_df['Home'].astype(str).str.replace('\xa0', ' ', regex=False).str.strip()
schedule_df['Away'] = schedule_df['Away'].astype(str).str.replace('\xa0', ' ', regex=False).str.strip()

# --- Detect week column ---
week_col_candidates = [col for col in schedule_df.columns if 'week' in col.lower()]
if not week_col_candidates:
    raise ValueError("No column found containing 'week'")
week_col = week_col_candidates[0]
weeks = sorted(schedule_df[week_col].unique())

matrix_df.index = matrix_df.index.astype(str).str.replace('\xa0', ' ', regex=False).str.strip()

# Ensure 'Adjusted Win Total' exists
if 'Adjusted Win Total' not in matrix_df.columns:
    raise ValueError("Matrix dataframe does not contain 'Adjusted Win Total' column.")

teams = sorted(matrix_df.index)
adjusted_win_total = matrix_df['Adjusted Win Total'].to_dict()

# --- Initialize Overall Ratings ---
overall_ratings = adjusted_win_total.copy()
print(sorted(overall_ratings.keys()))

# --- Iterative update ---
n_iter = 20

for iteration in range(n_iter):
    new_ratings = {}
    for team in teams:
        # Find all games this team played
        games = schedule_df[(schedule_df['Home'] == team) | (schedule_df['Away'] == team)]
        opponents = []
        for _, game in games.iterrows():
            opponent = game['Away'] if game['Home'] == team else game['Home']
            opponents.append(overall_ratings.get(opponent, 0))
        
        # Avoid division by zero
        avg_opponent_rating = sum(opponents) / len(opponents) if opponents else 0
        
        # Update Overall Rating
        new_ratings[team] = adjusted_win_total[team] + avg_opponent_rating
    
    overall_ratings = new_ratings.copy()  # update for next iteration
    print(f"Iteration {iteration}: {overall_ratings.get('Alabama')}")

# --- Compute Wins and Losses for each team ---
wins = {team: 0 for team in teams}
losses = {team: 0 for team in teams}

for _, game in schedule_df.iterrows():
    home = game['Home']
    away = game['Away']
    diff = game['Score_Differential']  # negative → away win

    if pd.isna(diff):
        continue  # skip games without results

    if diff < 0:  # away team wins
        wins[away] += 1
        losses[home] += 1
    else:         # home team wins
        wins[home] += 1
        losses[away] += 1

# --- Build final sorted rating table ---
team_table = (
    pd.DataFrame({
        'Team Name': list(overall_ratings.keys()),
        'Overall Rating': list(overall_ratings.values()),
        'Wins': [wins[t] for t in overall_ratings.keys()],
        'Losses': [losses[t] for t in overall_ratings.keys()]
    })
    .sort_values(by='Overall Rating', ascending=False)
    .reset_index(drop=True)
)

# Add Rank (1 = highest rating)
team_table['Rank'] = team_table.index + 1

# Reorder columns
team_table = team_table[['Rank', 'Team Name', 'Overall Rating', 'Wins', 'Losses']]

# --- Save to Excel ---
team_table.to_excel(final_team_rankings_output_excel_path, index=False)

print(f"Overall ratings saved to '{final_team_rankings_output_excel_path}'")


# In[ ]:


### Optional output Excels for further details on simulated season results below:


# In[12]:


# Schedule Output - each game, the projected outcome, and the actual outcome
schedule_df.to_excel(schedule_excel_file_path, index=False)

print(f"Excel file '{schedule_excel_file_path}' has been created successfully!")


# In[13]:


# Matrix Output - contains a matrix with every team on X-axis and Y-axis, and the total adjusted wins each team has against each opponent
matrix_df.to_excel(matrix_excel_file_path)

print(f"Excel file '{matrix_excel_file_path}' has been created.")


# In[20]:


# Generate output Excel with ratings for all opponents of each team

# --- Detect week column ---
week_col_candidates = [col for col in schedule_df.columns if 'week' in col.lower()]
if not week_col_candidates:
    raise ValueError("No column found containing 'week'")
week_col = week_col_candidates[0]
weeks = sorted(schedule_df[week_col].unique())

# --- Build lookup from team_table ---
rating_lookup = dict(zip(team_table["Team Name"], team_table["Overall Rating"]))

# Use the same team order as overall_ratings
teams = list(overall_ratings.keys())

# --- Build opponent table ---
team_rows = []

for team in teams:
    row = {'Team Name': team}
    
    for week in weeks:
        week_match = schedule_df[
            (schedule_df[week_col] == week) &
            ((schedule_df['Home'] == team) | (schedule_df['Away'] == team))
        ]
        
        if not week_match.empty:
            opponent = (
                week_match.iloc[0]['Away']
                if week_match.iloc[0]['Home'] == team
                else week_match.iloc[0]['Home']
            )
            opponent_rating = rating_lookup.get(opponent, None)
        else:
            opponent = None
            opponent_rating = None
        
        row[f'Week {week} Opponent'] = opponent
        row[f'Week {week} Opponent Rating'] = opponent_rating
    
    team_rows.append(row)

# Convert to DataFrame
opponent_table = pd.DataFrame(team_rows)

# --- Add Average Opponent Rating column ---
rating_cols = [col for col in opponent_table.columns if 'Opponent Rating' in col]
opponent_table['Average Opponent Rating'] = opponent_table[rating_cols].mean(axis=1)

# --- Save to Excel ---
opponent_table.to_excel(opponent_details_excel_path, index=False)

print(f"New table with opponents and ratings saved to '{opponent_details_excel_path}'")


# In[16]:


def get_team_results(team_name: str, schedule_df: pd.DataFrame, team_table: pd.DataFrame) -> pd.DataFrame:
    
    # Filter for all games involving the team
    team_games = schedule_df[
        (schedule_df["Home"] == team_name) | (schedule_df["Away"] == team_name)
    ].copy()
    
    if team_games.empty:
        raise ValueError(f"No games found for team '{team_name}'.")

    # Home/Away detection
    team_games["Home/Away"] = team_games.apply(
        lambda row: "Home" if row["Home"] == team_name else "Away",
        axis=1
    )

    # Opponent name
    team_games["Opponent"] = team_games.apply(
        lambda row: row["Away"] if row["Home"] == team_name else row["Home"],
        axis=1
    )

    # Determine W/L based on score differential
    def determine_result(row):
        if row["Home/Away"] == "Home":
            return "W" if row["Score_Differential"] > 0 else "L"
        else:
            return "W" if row["Score_Differential"] < 0 else "L"

    team_games["Result"] = team_games.apply(determine_result, axis=1)

    # FIX: team_table uses "Team Name", not "Team"
    opponent_info = team_table.rename(columns={"Team Name": "Opponent"})

    # Merge opponent rank & rating
    team_games = team_games.merge(opponent_info, on="Opponent", how="left")

    # Select final output
    final_table = team_games[[
        "Week",
        "Opponent",
        "Rank",
        "Overall Rating",
        "Home/Away",
        "Result"
    ]].sort_values("Week")

    # Rename columns for clarity
    final_table = final_table.rename(columns={
        "Rank": "Opponent Rank",
        "Overall Rating": "Opponent Rating"
    })

    return final_table


results = get_team_results("Notre Dame", schedule_df, team_table)
print(results)


# In[18]:


# Step 1: Filter top 12 teams from team_table
top_teams = team_table[team_table['Rank'].between(1, 12)].copy()

# Step 2: Rename columns
top_teams.rename(columns={'Rank': 'Seed', 'Team Name': 'Team'}, inplace=True)

# Step 3: Strip whitespace to improve matching
df['Team'] = df['Team'].str.strip()
top_teams['Team'] = top_teams['Team'].str.strip()

# Step 4: Map Rating from df to top_teams
rating_map = df.set_index('Team')['Rating']
top_teams['Rating'] = top_teams['Team'].map(rating_map)

# Step 5: Keep only the desired columns
final_output = top_teams[['Seed', 'Team', 'Rating']].copy()

# Step 6: Export to CSV
final_output.to_csv(tournament_data_csv_path, index=False)

print(f"Export complete: '{tournament_data_csv_path}'")

