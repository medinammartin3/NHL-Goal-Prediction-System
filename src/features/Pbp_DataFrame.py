
import os
import sys
sys.path.append(os.path.abspath("../"))
from src.GamesUtils import GamesUtils
from src.data.Data import Data
import pandas as pd
import math

class Pbp_to_DataFrame:

    def __init__(self):
        # Get the absolute path to the project root and then to games_data
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        self.data_path = os.path.join(project_root, 'games_data')

        
    def get_game(self, id):
        """Load a specific game play-by-play JSON file 

        Args:
            id (String): Game ID

        Returns:
            dic: dictionnary of the JSON file
        """
        season = id[:4] + '-' + str(int(id[:4])+1)
        game = Data.get_data(os.path.join(self.data_path, season, id+'.json'))
        return game
     

    def get_net_and_situation(self, teams, details, event):
        """Determine the net state (empty or not) and the game situation 
            (even strength, power play, short-handed)


        Args:
            teams (dict): Dictionary mapping team ID to team name and home/away team
            details (dict): Details of certain event in a game 
            event (dict): The event itself

        Returns:
            Tuple(String, String): Tuple containing the net situation (filet) and the game situation (even strength, power play, short-handed)
        """
        team_event_type = teams.get(details['eventOwnerTeamId'])[1]
        situationCode = event.get('situationCode')
        if situationCode is None:
            away_goalie = away_player = home_player = home_goalie = None
        else:
            away_goalie = int(situationCode[0])
            away_player = int(situationCode[1])
            home_player = int(situationCode[2])
            home_goalie = int(situationCode[3])

        # Home team
        if team_event_type == 'home':
            if away_goalie == 1:
                filet = 'Not empty'
            elif away_goalie == 0:
                filet = 'Empty'
            else:
                filet = None
                
            if None in (away_goalie, away_player, home_goalie, home_player):
                situation = 'Unknown'
            else:
                if (away_goalie + away_player) == (home_goalie + home_player):
                    situation = 'Even strength'
                elif (away_goalie + away_player) > (home_goalie + home_player):
                    situation = 'Short-handed'
                elif (away_goalie + away_player) < (home_goalie + home_player):
                    situation = 'Power play'
                else:
                    situation = None
            
        # Away team
        elif team_event_type == 'away':
            if home_goalie == 1:
                filet = 'Not empty'
            elif home_goalie == 0:
                filet = 'Empty'
            else:
                filet = None
                
            if None in (away_goalie, away_player, home_goalie, home_player):
                situation = 'Unknown'
            else:
                if (away_goalie + away_player) == (home_goalie + home_player):
                    situation = 'Even strength'
                elif (away_goalie + away_player) > (home_goalie + home_player):
                    situation = 'Power play'
                elif (away_goalie + away_player) < (home_goalie + home_player):
                    situation = 'Short-handed'
                else:
                    situation = None
        return filet, situation, situationCode

    def get_event_type_and_player(self, roaster, details, event):
        """Get the event type (Shot/Goal) and the player involved

        Args:
            roaster (dict): Dictionary mapping each player id the their name
            details (dict): Details of certain event in a game
            event (dict): The event itself

        Returns:
            Tuple(Sting, String): Tuple containing the name of the involved player and the event type
        """
        if event['typeDescKey'] == 'shot-on-goal':
            player_id = details.get('shootingPlayerId')
            player = roaster.get(player_id, [None])[0] if player_id else None
            event_type = 'Shot'
        elif event['typeDescKey'] == 'goal':
            player = roaster.get(details['scoringPlayerId'])[0]
            event_type = 'Goal'
        else:
            player = None
            event_type = None

        return player, event_type
    
    def get_home_team(self, teams):
        for team in teams.values():
            if team[1] == "home":
                home_team_name, _ = team
                break
        return home_team_name.split(" - ")[0]
    
    def build_game_DataFrame(self, game_id, game=None):
        """Build a DataFrame of shot and goal events for a specific game

        Args:
            game_id (String): The game ID for wich we want to build the DataFrame

        Returns:
            pandas.Dataframe: DataFrame with one row per shot or goal event, containing:
                - Game ID: ID of the game where the event took place (int)
                - ID (int) : Event ID
                - Team (String): Team abbreviation
                - Event Type (String): 'Shot' or 'Goal'
                - Type of Shot (String): Shot type (e.g., 'wrist', 'slap', 'Unknown')
                - Shooter (String or None): Name of the shooting/scoring player
                - Goalie (String or None): Name of the goalie in net
                - Net (String or None): 'Not empty', 'Empty', or None
                - Situation (String or None): 'Even strength', 'Power play', 'Short-handed'.
                - Period (int): Game period number
                - Time (String): Time in the period (MM:SS format)
                - X (int or None): X coordinate of the event
                - Y (int or None): Y coordinate of the event
                - Zone (String): The zone of the event ("O", "D" or "N")
                - Home Team D Side (String): The defending side of the home team
                - Home Team (String) : The name of the home team
                - Last Event Type (String or None): The type of the event immediately preceding this event.
                - Last X (int or None): X coordinate of the previous event if available.
                - Last Y (int or None): Y coordinate of the previous event if available.
                - Time Since Last (int or None): Time elapsed in seconds between the current and previous event.
                - Distance From Last (float or None): Euclidean distance between the coordinates of the current and previous event.

        """
        #away goalie (1=in net, 0=pulled)-away skaters-home skaters-home goalie (1=in net, 0=pulled)
        #https://gitlab.com/dword4/nhlapi/-/issues/112?
        if game is None:
            game = self.get_game(game_id)
        teams = GamesUtils.get_teams(game)
        roaster = GamesUtils.get_game_roaster(game)
        home_team = self.get_home_team(teams)
        game_type_code = game.get('gameType')
        if game_type_code == 2:
            game_type = "Regular"
        elif game_type_code == 3:
            game_type = "Playoffs"
        else:
            game_type = "Other"

        pbp_df = []
        for index, event in enumerate(game['plays']):
            if event['typeDescKey'] == 'shot-on-goal' or event['typeDescKey'] == 'goal':
                details = event['details']
            
                event_id = event['eventId']
                event_team = teams.get(details['eventOwnerTeamId'])[0].split('-')[0]
                goalie_id = details.get('goalieInNetId')
                goalie = roaster.get(goalie_id, [None])[0] if goalie_id else None
                period = event['periodDescriptor']['number']
                period_time = event['timeInPeriod']
                x = details.get('xCoord')
                y = details.get('yCoord')
                shotType = details.get('shotType', 'Unknown')
                shot_zone = details.get('zoneCode')
                home_team_D_side = event.get('homeTeamDefendingSide')

                player, event_type = self.get_event_type_and_player(roaster, details, event)
                filet, situation, situationCode = self.get_net_and_situation(teams, details, event)

                """
                Last event information
                """
                # First event on the game so no last event
                if index == 0:
                    last_event_type = None
                    last_event_x = None
                    last_event_y = None
                    time_diff = None
                    last_event_distance = None
                else:
                    # Find last event
                    last_event = game['plays'][index-1]
                    last_event_details = last_event.get('details', {})

                    # Type
                    last_event_type = last_event.get('typeDescKey')

                    # Coords
                    last_event_x = last_event_details.get('xCoord')
                    last_event_y = last_event_details.get('yCoord')

                    # Time
                    last_event_period = last_event['periodDescriptor']['number']
                    last_event_period_time = last_event['timeInPeriod']

                    min_curent_event, sec_curent_event = map(int, period_time.split(':'))
                    min_last_event, sec_last_event = map(int, last_event_period_time.split(':'))

                    # Time from start of the game:
                    # --> 1 period = 20 min = 1200 sec
                    # --> Time_from_start (in seconds) = periods_completed * 1200 + minutes_current_period * 60 + seconds_current_period
                    game_time_current_event = (period - 1) * 1200 + min_curent_event * 60 + sec_curent_event
                    game_time_last_event = (last_event_period - 1) * 1200 + min_last_event * 60 + sec_last_event

                    # Time difference
                    time_diff = game_time_current_event - game_time_last_event

                    # Distance
                    coords_list = [x, last_event_x, y, last_event_y]
                    if None not in coords_list:
                        last_event_distance = math.sqrt((x - last_event_x) ** 2 + (y - last_event_y) ** 2)
                    else:
                        last_event_distance = None

                row = {'Game ID': game_id,
                       'Game Type': game_type,
                       'ID': event_id, 
                       'Team':event_team, 
                       'Event Type':event_type, 
                       'Type of Shot':shotType, 
                       'Shooter':player, 
                       'Goalie':goalie,
                       'Net':filet,
                       'Situation':situation,
                       'situationCode': situationCode,
                       'Period':period,
                       'Time':period_time,
                       'X':x,
                       'Y':y,
                       'Zone': shot_zone,
                       'Home Team D Side': home_team_D_side,
                       'Home Team': home_team,
                       'last_event_type': last_event_type,
                       'last_x': last_event_x,
                       'last_y': last_event_y,
                       'time_since_last': time_diff,
                       'distance_from_last': last_event_distance
                    }

                pbp_df.append(row)

        return pd.DataFrame(pbp_df)


if __name__=="__main__":
    c = Pbp_to_DataFrame()
    df = c.build_game_DataFrame('2023020727')
    print(df)
    shots = df[df['Event Type'] == 'Goal']
    print(shots)
    pt = shots.pivot_table(index='Team',
                       columns='Type of Shot',
                       values='ID',
                       aggfunc='count',
                       fill_value=0)
    print(pt)
