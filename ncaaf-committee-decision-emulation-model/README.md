\# NCAAF Committee Decision Emulation Model

\#\# Disclaimer on Usage of Generative AI  
While this work was created by me, the code itself was largely generated through prompting GenAI (ChatGPT). I provided the prompts and any necessary code debugging throughout the process \- but it was artificial intelligence that was the primary developer for the Python script.

\#\# Overview  
This code first inputs a list of Division 1 NCAAF teams and their predictive ratings from [TeamRankings.com](http://TeamRankings.com). Then, it generates a randomized schedule with random pairings of teams, separated into different "weeks" of the season. Next, it simulates the full season using the randomized schedule by running a random normal distribution for each game. Finally, it uses the iteratively reweighted least squares method to calculate final ratings for each team and outputs, in an attempt to emulate the human decision-making from the NCAAF ranking committee.

\#\# Libraries  
pandas, numpy, and random

\#\# Input  
The input is created by copying and pasting the table from the NCAAF \- Predictive Ratings page on [TeamRankings.com](http://TeamRankings.com), then reducing it to only the Rating column and the Team column (with win-loss record removed from the team name). This is then exported as CSV to be used as input.

\#\# Main Output  
The main output for this code contains the final team rankings that are generated after using the iteratively reweighted least squares method. For each team, it also contains their final rating, as well as their wins and losses.

\#\# Optional Outputs  
There are a set of optional outputs to generate for further details into the simulated season. The schedule\_excel\_file\_path output provides every game in the season as a row, with projected outcome and actual score differential. The matrix\_excel\_file\_path output creates a matrix with every team, to show the final win-loss results between each team (a cell contains the record between the two teams in the X and Y axes). The opponent\_details\_excel\_path provides the final rating of each team’s opponents throughout the season, listed by week. Finally, the tournament\_data\_csv\_path generates a CSV of the top twelve ranked teams (from iteratively reweighted least squares) and their TeamRankings Predictive Rating. This is generated in a format so that it can be used as an input to my NCAAF Tournament Simulator script in Python.

\#\# Background  
I was watching the college football bowl games during the week of Christmas 2022, and I started to develop a curiosity. I knew that, to assign teams to the NCAAF “playoff”, there was a human committee that ranked the teams. I also knew that teams’ schedules varied significantly from each other, because there was no standardization \- teams set up their own schedules by making agreements with other teams. I began to wonder how the committee would approach such a complex and open-ended ranking exercise.

I already was familiar with the concept of using random normal distributions to simulate individual sports events, so I realized I could group many of these together to simulate a full hypothetical season. This would then give me the data I needed to determine the ideal mathematical method for ranking.

\#\# Finding the Ideal Mathematical Ranking Method  
My first attempt to find a mathematical method for ranking these teams turned to an old-fashioned metric \- RPI. RPI takes the winning percentage for a team, for all the teams they played, and for all the teams those teams played, then weights the percentages to get a single rating per team.

That didn’t look right with my fictional season results, though \- the teams being ranked highly didn’t pass my subjective test of look and feel. I scrolled the Internet for other athletic ranking methods.

Finally, I stumbled upon mathematician Kenneth Massey’s “Massey Rating”, and the method of iteratively reweighted least squares. The only way I could realistically emulate human-level decision making in this case, I found, was to allow the algorithm to iterate on itself. The ranking math worked like this \- first, you needed to take the number of wins for each team as the team’s initial rating (with slight adjustments for home-field advantage). Then, you’d find the average rating for every team they played against. Now, you could adjust the team rating for schedule difficulty by adding those two numbers together. But, here’s the catch \- by adjusting the teams’ ratings, suddenly the average rating of the teams they played was now different, changing the team’s rating itself. This creates an iterative loop, where team ratings (number of wins plus average opponent rating) are updated, which then changes their average opponent rating, necessitating recalculation \- over and over, until the rating converges.

\#\# Additional Assumptions and Considerations  
Home Field Advantage:   
Home-field advantage is taken into account for both the game simulations, as well as the final team ranking calculation. For the game simulations, I adjusted the input “spreads” (used as the mean for each random normal distribution) by 2.6 points in favor of the home team, as taken from [TeamRankings.com](http://TeamRankings.com). 

For the final team ranking calculation, each away win is counted as 1.125 wins for a team, while each home win is counted as 0.875 wins (and vice-versa for losses). If Georgia beats Alabama in Georgia, it is not as impressive to the human committee compared to Georgia going into Alabama and beating them on their home turf. Equally, it is a worse sign for Alabama’s skill if they lose in Alabama, rather than losing in Georgia. 

1.125 and 0.875 were subjective numbers I decided on when looking at the win probability of home teams, and thinking about the corresponding Z-score.

Score Differential:   
The score differential of games is not taken into account for the final team ranking calculation. While I typically would consider this, it is my impression that the primary consideration of the  NCAAF committee in real life is if a game is simply a win or a loss, regardless of whether that win is by one point or one hundred points.

Random Normal Distribution Inputs:

Each "game" was simulated with a single random normal distribution, where the mean was the projected score differential (Home Team Predictive Rating - Away Team Predictive Rating + 2.6), and the standard deviation was 13.89. I do not remember exactly where I first found this 13.89 number, but it may have been from this article, which cites the same standard deviation for NCAAF games - https://www.footballperspective.com/a-monte-carlo-based-comparison-of-college-football-playoff-systems/.
