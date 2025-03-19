"""
This script contains data retrieving functions for my Tennis_Predictor
project. It should be working independently of machine and location within it
"""

import os
import numpy as np
import pandas as pd

class Tennis:
    def get_singles(self) -> dict:
        """
        Returns a dictionary whose keys are the years of the available data
        and whose values are dataframes containing the data of each year's
        atp singles matches.
        """

        # finding root directory
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # defining absolute path to the dir where the data is
        csv_path = os.path.join(root_dir, "raw_data", "tennis_atp", "singles")

        # creating dictionary
        file_names = [name for name in os.listdir(csv_path) if name.endswith(".csv")]

        years_dict = {}

        for file in file_names:
            year = int(file.split("atp_matches_")[1].split(".csv")[0])
            years_dict[year] = pd.read_csv(os.path.join(csv_path, file))

        return years_dict


    def get_doubles(self) -> dict:
        """
        Returns a dictionary whose keys are the years of the available data
        and whose values are dataframes containing the data of each year's
        atp doubles matches.
        """

        # finding root directory
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # defining absolute path to the dir where the data is
        csv_path = os.path.join(root_dir, "raw_data", "tennis_atp", "doubles")

        # creating dictionary
        file_names = [name for name in os.listdir(csv_path) if name.endswith(".csv")]

        years_dict = {}

        for file in file_names:
            year = int(file.split("atp_matches_doubles_")[1].split(".csv")[0])
            years_dict[year] = pd.read_csv(os.path.join(csv_path, file))

        return years_dict


    def get_players(self) -> pd.DataFrame:
        """
        Returns a dataframe with the unique ID of each player as index.
        Includes names, date of birth, country of origin, and hand, as well as
        height and wikipedia ID.
        """

        # finding root directory
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # defining absolute path to the dir where the data is
        csv_path = os.path.join(root_dir, "raw_data", "tennis_atp", "players", \
            "atp_players.csv")

        return pd.read_csv(csv_path)

    def get_rankings(self) -> dict:
        """
        Returns a dictionary with the rankings for a whole decade, with the
        exception of the current decade. The keys are the last two digits of
        the decade, and the values are the dataframes, again with the exception
        of the current rankings whose key is the string 'current'. The index of
        the dataframe is the ranking date and it also contains the ID of the
        player, the rank and the points they had at that time. The decades
        spanned are from the 1970s up to late May 2024.
        """

        # finding root directory
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # defining absolute path to the dir where the data is
        csv_path = os.path.join(root_dir, "raw_data", "tennis_atp", "rankings")

        # creating dictionary
        file_names = [name for name in os.listdir(csv_path) if name.endswith(".csv")]

        decade_dict = {}

        for file in file_names:
            if file.endswith("current.csv"):
                decade = "current"
                decade_dict[decade] = pd.read_csv(os.path.join(csv_path, file))
            else:
                decade = int(file.split("atp_rankings_")[1].split("s.csv")[0])
                decade_dict[decade] = pd.read_csv(os.path.join(csv_path, file))

        return decade_dict
