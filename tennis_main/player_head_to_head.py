"""
This module contains methods that create and extract the head to head statistics
between players.
"""

# general imports
import pandas as pd
import numpy as np
import os

# project related imports
from tennis_main.data import Tennis
from tennis_main.player_stats_new import Player
from tennis_main.utilities import score_parser

# for sliding window rolling method
from collections import deque

class Player_H2H(Player):
    """
    Child class of Player class, that inherits all the stat calculating methods from the parent class.
    """

    def __init__(self):
        super().__init__()

    #################
    # h2h dataframe #
    #################

    def get_h2h_matches(self, player1_id: int, player2_id: int, df: pd.DataFrame) -> pd.DataFrame:
        """
        Retrieves all head-to-head matches between two players from a DataFrame.
        Uses parent class's methods to create explicit stats and also offset the dates
        for subsequent h2h stat calculations.
        """
        h2h_mask = ((df["winner_id"] == player1_id) & (df["loser_id"] == player2_id)) | \
                   ((df["winner_id"] == player2_id) & (df["loser_id"] == player1_id))

        h2h_df = df[h2h_mask].copy()

        h2h_df = self.date_offset(h2h_df)
        h2h_df = self.create_basic_features(h2h_df)

        h2h_df.sort_values(by='adjusted_date', inplace=True)

        return h2h_df

    def calculate_h2h_wins(self, player1_id: int, player2_id: int, h2h_df: pd.DataFrame) -> tuple:
        """
        Calculates the number of wins for each player in their head-to-head matches.
        Returns two tuples, in the form (player_id, wins).
        """
        h2h_df = h2h_df.copy()

        player1_wins = h2h_df[h2h_df["winner_id"] == player1_id].shape[0]
        player2_wins = h2h_df[h2h_df["winner_id"] == player2_id].shape[0]

        return (player1_id, player1_wins), (player2_id, player2_wins)


    def get_h2h_cumul_stats(self, player1_id: int, player2_id: int, h2h_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates cumulative head-to-head statistics up to the current match between two players.
        Does so for all surfaces and on a per surface basis.
        """
        h2h_df = h2h_df.copy()

        cumul_stats = {}

        # all surfaces
        for stat in self.STATS_ALREADY_THERE+ self.STATS_EXPLICITLY_CREATED:
            w_stat_col = f"w_{stat}"
            l_stat_col = f"l_{stat}"

            player1_stat_this_match = (h2h_df["winner_id"] == player1_id) * h2h_df[w_stat_col] + (h2h_df["loser_id"] == player1_id) * h2h_df[l_stat_col]
            player2_stat_this_match = (h2h_df["winner_id"] == player2_id) * h2h_df[w_stat_col] + (h2h_df["loser_id"] == player2_id) * h2h_df[l_stat_col]

            cumul_stat_p1 = player1_stat_this_match.cumsum()
            cumul_stat_p2 = player2_stat_this_match.cumsum()

            cumul_stats[f"cumul_{player1_id}_{stat}"] = cumul_stat_p1
            cumul_stats[f"cumul_{player2_id}_{stat}"] = cumul_stat_p2

            # per surface
            for s in self.SURFACES:
                cumul_stat_p1_surface = (player1_stat_this_match * (h2h_df["surface"] == s)).cumsum()
                cumul_stat_p2_surface = (player2_stat_this_match * (h2h_df["surface"] == s)).cumsum()

                cumul_stats[f"cumul_{player1_id}_{stat}_{s.lower()}"] = cumul_stat_p1_surface
                cumul_stats[f"cumul_{player2_id}_{stat}_{s.lower()}"] = cumul_stat_p2_surface

        h2h_df = h2h_df.assign(**cumul_stats).copy()

        return h2h_df

    def get_h2h_advanced_ratio_stats(self, player1_id: int, player2_id: int, h2h_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates advanced ratio statistics up to the current match between two players.
        """
        h2h_df = h2h_df.copy()

        for player in [player1_id, player2_id]:
            advanced_ratios_dict = {f"{player}_1st_serve_won_ratio": (f"cumul_{player}_1stWon", f"cumul_{player}_1stIn"),
                                    f"{player}_2nd_serve_won_ratio": (f"cumul_{player}_2ndWon", f"cumul_{player}_2ndIn"),
                                    f"{player}_ace_p_point_ratio": (f"cumul_{player}_ace", f"cumul_{player}_svpt"),
                                    f"{player}_df_p_point_ratio": (f"cumul_{player}_df", f"cumul_{player}_svpt"),
                                    f"{player}_bp_saved_ratio": (f"cumul_{player}_bpSaved", f"cumul_{player}_bpFaced"),
                                    f"{player}_bp_conversion_ratio": (f"cumul_{player}_bp_won", f"cumul_{player}_bp_caused"),
                                    f"{player}_break_ratio": (f"cumul_{player}_bp_won", f"cumul_{player}_served_games_against_player"),
                                    f"{player}_hold_ratio": (f"cumul_{player}_serve_games_held", f"cumul_{player}_SvGms"),
                                    f"{player}_tiebreaks_won_ratio": (f"cumul_{player}_tiebreaks_won", f"cumul_{player}_tiebreaks_happened"),
                                    f"{player}_tiebreaks_lost_ratio": (f"cumul_{player}_tiebreaks_lost", f"cumul_{player}_tiebreaks_happened")}

            for stat, (numerator_col, denominator_col) in advanced_ratios_dict.items():
                h2h_df = self._calculate_advanced_ratio_statistics(h2h_df, numerator_col, denominator_col, stat)

        return h2h_df
