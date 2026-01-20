import shutil
from concurrent.futures.thread import ThreadPoolExecutor
import concurrent.futures as cf

import requests
import json
import os
import time #for sleep time

class Data:
    def __init__(self):


        self.data_path = '../../games_data'

        os.makedirs(self.data_path, exist_ok=True)

        self.session = requests.Session()
        self.max_workers = 1
        self.data = self.get_all_games_id()
        self.big_file_path = 'play_by_play.json'

    def __add__(self, season):
        """ Add a season to the dataset via the `+` operator.

        Args:
            season (String): Season to add in the format "YYYY-YYYY"

        Raises:
            TypeError: If `season` is not a string.

        Returns:
            self: The current instance with the new season's data loaded.
        """
        if not isinstance(season, str):
            raise TypeError("Season must be a string of format YYYY-YYYY")

        self.load_data_local(season)
        return self

    def get_all_games_id(self):
        """Retrieve all game IDs from the NHL API or local cache.

        Raises:
            RuntimeError:  If the API request fails

        Returns:
            dic : JSON data containing all IDs of the games
        """
        file_path = os.path.join(self.data_path, 'all_games.json')
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                all_data = json.load(file)
        else:
            response = self.session.get("https://api.nhle.com/stats/rest/en/game")
            if response.status_code == 200:
                all_data = response.json()
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(all_data, f, ensure_ascii=False, indent=4)
            else:
                raise RuntimeError(response.status_code)
        return all_data


    def get_games_id_from_season(self, season_start):
        """ Get all games IDs from a specific season

        Iterate over the loaded game data and extract IDs of the season games.

        Args:
            season_start (String): THe start year of the season for example "2016-2017" -> "2016"

        Returns:
            List[int]: List fo all the season games IDs
        """
        id_list_game = []

        for element in self.data['data']:
            if str(element['id'])[:4] in season_start:
                id_list_game.append(element['id'])
        return id_list_game

    def fetch_one_game_pbp(self, game_id):
        """Get the play-by-play info of a specific game

        Args:
            game_id (int): The gaem ID we want to retrieve the play-by-play from.

        Returns:
            Tuple: Tuple containing the json fo the play-by-play data or if absent return None,
                    game_id,
                    None or the exception if the request to the api fail.
        """
        try:
            pbp = self.session.get(f"https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play")
            if pbp.status_code == 404:
                return None, game_id, 404
            elif pbp.status_code == 429:
                print(f'Rate limit hit for game {game_id}')
                time.sleep(2)
                pbp = self.session.get(f"https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play")
            pbp.raise_for_status()
            return pbp.json(), game_id, None
        except requests.exceptions.RequestException as e:
            return None, game_id, e


    def load_data_local(self, season, merge_one_file=True):
        """Load play-by-play data for one or more seasons into the local data folder.

            For each season provided:
                - Retrieve all games IDs for the season.
                - Checks wich games are already downloaded in the local data.
                - Retrieve missing games data for play-plby-play.
                - Saves each game JSON file in a season folder.
                - Optionally merges all ames play-by-play  for th seasons into one big JSON file.

        Args:
            season (String or List[String]): The season or the seasons list to be load.
            merge_one_file (bool, optional): If true also merge all the games into one big file. Defaults to True.

        Raises:
            TypeError: If `season` is neither a string nor a list of strings.
        """

        if isinstance(season, str):
            season = [season]
        elif isinstance(season, list):
            season = season
        else:
            raise TypeError("Season must be string or list")

        pbp_big_file = None
        for s in season:

            season_start = s.split("-")[0]
            id_list_games = self.get_games_id_from_season(season_start)
            total_games = len(id_list_games)
            futures = []
            season_dir = os.path.join(self.data_path, s)
            os.makedirs(season_dir, exist_ok=True)

            games_not_fetch = []
            for gid in id_list_games:
                out_path = os.path.join(season_dir, f"{gid}.json")
                if not os.path.exists(out_path):
                    games_not_fetch.append(gid)

            print(f"=== Fetching {len(games_not_fetch)} games from season {s} ===")
            if not games_not_fetch:
                print(f"Season {s} already load in data folder")
                continue

            play_by_play_result = {}
            with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
                count = 0
                for ids in games_not_fetch:
                    futures.append(ex.submit(self.fetch_one_game_pbp, ids))

                for i, completed in enumerate(cf.as_completed(futures)):
                    pbp, game_id, e = completed.result()

                    if e is None and pbp is not None:
                        out_path = os.path.join(season_dir, f'{game_id}.json')
                        Data.save_json(pbp, out_path)

                        if merge_one_file:
                            play_by_play_result[game_id] = pbp

                    count = count + 1
                    if count%50 == 0 or count == total_games:
                        print(f"{count}/{total_games} games fetched")

            if merge_one_file and play_by_play_result:
                self.add_data_to_big_file(pbp_big_file, s, play_by_play_result)


    def add_data_to_big_file(self, pbp_big_file, s, play_by_play_result):
        """Add data to the big data file containing all the games Pbp.

        Args:
            pbp_big_file (dic): The file we want to add the data to
            s (String): The season
            play_by_play_result (dic): The play-by-play information of all games in a season
        """
        if pbp_big_file is None:
            pbp_big_file = self.load_big_pbp_file()
        if s not in pbp_big_file:
            pbp_big_file[s] = {}

        pbp_big_file[s].update(play_by_play_result)
        Data.save_json(pbp_big_file, os.path.join(self.data_path, self.big_file_path))

    def load_big_pbp_file(self):
        """Load the big file containing all the games Pbp data

        Returns:
            dic: The big file loaded
        """
        path = os.path.join(self.data_path, self.big_file_path)
        os.makedirs(self.data_path, exist_ok=True)
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
        return {}

    @staticmethod
    def get_data(file_path):
        """Load the file (data)

        Args:
            file_path (String): The file path we want to load.

        Raises:
            ValueError: If the file doesn't exist

        Returns:
            dic: The JSON file loaded
        """
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                file = json.load(file)
            return file
        else:
            raise ValueError("File path doesnt exist")

    @staticmethod
    def save_json(data_file, file_path):
        """Save the data into a json format

        Args:
            data_file (dic): The data we want to save
            file_path (String): THe path we want to save the data
        """
        with open(file_path, "w") as f:
            json.dump(data_file, f)

    def remove_season(self, season):
        """Remove a season from the data in the season folder and in the big data file.

        Args:
            season (String): The season we want to delete

        Raises:
            ValueError: if the big file is not found
        """
        season_path = os.path.join(self.data_path, season)
        big_file_path = os.path.join(self.data_path, self.big_file_path)

        if os.path.exists(season_path):
            shutil.rmtree(season_path)


        if os.path.exists(big_file_path):
            with open(big_file_path, "r") as f:
                file = json.load(f)
        else:
            raise ValueError("File not found")

        if season in file:
            del file[season]
            Data.save_json(file, big_file_path)


if __name__=="__main__":
    d = Data()
    d.load_data_local(['2016-2017', '2017-2018', '2018-2019', 
                       '2019-2020', '2020-2021', '2021-2022', 
                       '2022-2023', '2023-2024',], merge_one_file=True)
    data = d.get_data(os.path.join(d.data_path, '2017-2018', '2017020005.json'))
    print(len(data))