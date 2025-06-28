"""
This module contains methods to extract per player statistics that will be
fed to the model.
"""
# general imports
import pandas as pd
import numpy as np
import os

# project related imports
from tennis_main.data import Tennis
from tennis_main.utilities import score_parser

# for sliding window rolling method
from collections import deque

class Player:
    """
    Class that is used to calculate the per player statistics
    on a match-by-match basis without paying particular attention
    to the opponent, and has a child class that is intended
    to calculate the head 2 head statistics separately.
    """

    SURFACES = ["Clay", "Grass", "Hard", "Carpet"]

    STATS_ALREADY_THERE = ["ace", "df", "svpt", "1stIn", "2ndIn", "1stWon",
                           "2ndWon", "bpFaced", "bpSaved", "SvGms"] # base names for w_ and l_ columns

    STATS_EXPLICITLY_CREATED = ["tiebreaks_won", "tiebreaks_lost", "tiebreaks_happened", # created names for w_ and l_ columns
                                "bp_caused", "bp_won", "served_games_against_player",
                                "serve_games_held"]

    # final player-specific column names after clean_player_data
    PLAYER_STATS = [f'player_{stat}' for stat in STATS_ALREADY_THERE] + \
                   [f'player_{stat}' for stat in STATS_EXPLICITLY_CREATED]

    # columns dropped by clean_player_data after player-specific stats are extracted
    COLS_TO_DROP_AFTER_CLEANING = ['winner_id', 'loser_id','winner_seed', 'winner_entry', 'winner_name', 'winner_hand', 'winner_ht',
                                   'winner_ioc', 'winner_age', 'loser_seed', 'loser_entry', 'loser_name',
                                   'loser_hand', 'loser_ht', 'loser_ioc', 'loser_age', 'score', 'best_of', 'round',
                                   'minutes', 'winner_rank', 'winner_rank_points', 'loser_rank', 'loser_rank_points',
                                   'tourney_name', 'tourney_level', 'draw_size']


    def __init__(self):
        self.singles_data = Tennis().get_singles()
        self.players_data = Tennis().get_players()

    def set_all_player_ids(self, df: pd.DataFrame) -> set:
        """
        Returns a set of all player_ids that are part of the dataframe passed.
        """
        return set(df["winner_id"].unique().tolist()+df["loser_id"].unique().tolist())

    def date_offset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Offsets match dates based on their original 'tourney_date' to create distinct 'adjusted_date'
        for matches occurring sequentially within tournaments.
        The input DataFrame should have 'tourney_date' and 'tourney_id' columns.
        """
        df = df.copy()

        # ensuring 'tourney_date' is datetime

        if not pd.api.types.is_datetime64_any_dtype(df["tourney_date"]):
            date_series_as_str = df["tourney_date"].astype(str)
            try:
                # parsing with YYYYMMDD format if it's integer/string
                df["tourney_date"] = pd.to_datetime(date_series_as_str, format='%Y%m%d')
            except ValueError:
                # pandas' general datetime parsing for other formats (e.g., YYYY-MM-DD)
                df["tourney_date"] = pd.to_datetime(date_series_as_str)

        # sorting before groupby to ensure cumcount is applied in a consistent order of matches
        sort_keys = ['tourney_date', 'tourney_id']

        if 'match_num' in df.columns:
            sort_keys.append('match_num')
            df.sort_values(by=sort_keys, inplace=True)
        else:
            # using original index if match_num is not there
            df = df.reset_index().rename(columns={'index': 'original_index_for_sort'})
            sort_keys.append('original_index_for_sort')
            df.sort_values(by=sort_keys, inplace=True)

            if 'original_index_for_sort' in df.columns: # remove if it was added
                 df.drop(columns=['original_index_for_sort'], inplace=True)

        # grouping by tourney_id and the date part of tourney_date to create the offset.
        df['date_offset_val'] = df.groupby(['tourney_id', df['tourney_date'].dt.floor('D')]).cumcount()

        df['adjusted_date'] = df['tourney_date'] + pd.to_timedelta(df['date_offset_val'], unit='D')

        df.drop(columns=['date_offset_val'], inplace=True)

        return df

    def find_player_rows(self, player_id: int, df: pd.DataFrame, surface: str = "A") -> pd.DataFrame:
        """
        Finds all entries of a given player id in a given df.
        Surface parameter can be passed. Possible values are:
        'A' -> All surfaces
        'H' -> Hard surface
        'C' -> Clay surface
        'Crp' -> Carpet surface
        'G' -> Grass surface
        """
        base_player_mask = (df["winner_id"] == player_id) | (df["loser_id"] == player_id)

        if surface == "A":
            final_mask = base_player_mask
        elif surface == "H":
            final_mask = base_player_mask & (df["surface"] == "Hard")
        elif surface == "C":
            final_mask = base_player_mask & (df["surface"] == "Clay")
        elif surface == "Crp":
            final_mask = base_player_mask & (df["surface"] == "Carpet")
        elif surface == "G":
            final_mask = base_player_mask & (df["surface"] == "Grass")
        else:
            raise ValueError(f"Invalid surface code: {surface}. Must be one of 'A', 'H', 'C', 'Crp', 'G'.")

        player_df = df[final_mask].copy()
        player_df.sort_values(by='adjusted_date', inplace=True)

        return player_df

    #################################################################################
    # creating basic features that are not explicit but are straightforward to find #
    #################################################################################

    def create_2nd_serve_in(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Method that creates an 2nd serve in feature, for later ease of use.
        """
        df = df.copy()

        w_2ndIn = df["w_svpt"] - df["w_1stIn"] - df["w_df"]
        l_2ndIn = df["l_svpt"] - df["l_1stIn"] - df["l_df"]

        df = df.assign(w_2ndIn=w_2ndIn, l_2ndIn=l_2ndIn)

        return df

    def create_tiebreaks(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Method that calculates the tiebreaks won and lost by the winner and loser in this match.
        """
        df = df.copy()

        w_tiebreaks_won, l_tiebreaks_won, tiebreaks_happened = zip(*df["score"].apply(score_parser))

        df = df.assign(w_tiebreaks_won = w_tiebreaks_won, \
                       w_tiebreaks_lost = l_tiebreaks_won, \
                       l_tiebreaks_won = l_tiebreaks_won, \
                       l_tiebreaks_lost = w_tiebreaks_won, \
                       w_tiebreaks_happened = tiebreaks_happened, \
                       l_tiebreaks_happened = tiebreaks_happened
                       )

        return df

    def create_bp_related_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Method that calculates various breakpoint related features for winner and loser.
        """
        df = df.copy()

        w_bp_caused = df["l_bpFaced"]
        l_bp_caused = df["w_bpFaced"]

        w_bp_won = df["l_bpFaced"] - df["l_bpSaved"]
        l_bp_won = df["w_bpFaced"] - df["w_bpSaved"]

        df = df.assign(w_bp_caused = w_bp_caused, l_bp_caused = l_bp_caused,
                       w_bp_won = w_bp_won, l_bp_won = l_bp_won)

        return df

    def create_games_served_against_player(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Method that calculates the games served against the winner and loser.
        """
        df = df.copy()

        w_served_games_against_player = df["l_SvGms"]
        l_served_games_against_player = df["w_SvGms"]

        df = df.assign(w_served_games_against_player = w_served_games_against_player,
                       l_served_games_against_player = l_served_games_against_player)

        return df

    def create_serve_games_held(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Method that calculates the games served by the winner and loser that were not broken.
        """
        df = df.copy()

        w_serve_games_held = df["w_SvGms"] - (df["w_bpFaced"] - df["w_bpSaved"])
        l_serve_games_held = df["l_SvGms"] - (df["l_bpFaced"] - df["l_bpSaved"])

        df = df.assign(w_serve_games_held = w_serve_games_held, l_serve_games_held = l_serve_games_held)

        return df

    def create_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies all new feature creation methods at once
        """
        df = df.copy()
        df = self.create_2nd_serve_in(df)
        df = self.create_tiebreaks(df)
        df = self.create_bp_related_features(df)
        df = self.create_games_served_against_player(df)
        df = self.create_serve_games_held(df)

        return df

    ####################################
    # calculations of cumulative stats #
    ####################################

    def find_cumul_stats(self, player_id: int, df: pd.DataFrame) -> pd.DataFrame:
        """
        Finds all basic cumulative stats for a given player efficiently using a single
        pass to create all new columns with df.assign().
        """
        df = df.copy()
        new_cols = {}

        all_stats = self.STATS_ALREADY_THERE + self.STATS_EXPLICITLY_CREATED

        for stat_prefix in all_stats:
            w_stat_col = f"w_{stat_prefix}"
            l_stat_col = f"l_{stat_prefix}"

            # fallback imputing if the master method has not been applied
            df[w_stat_col] = df[w_stat_col].fillna(0)
            df[l_stat_col] = df[l_stat_col].fillna(0)

            player_stat_this_match = (df["winner_id"] == player_id) * df[w_stat_col] + (df["loser_id"] == player_id) * df[l_stat_col]

            # all surfaces
            new_cols[f"cumul_{stat_prefix}"] = player_stat_this_match.cumsum().replace([np.inf, -np.inf], np.nan).fillna(0)

            # per surface
            for s in self.SURFACES:
                new_cols[f"cumul_{stat_prefix}_{s.lower()}"] = (player_stat_this_match * (df["surface"] == s)).cumsum().replace([np.inf, -np.inf], np.nan).fillna(0)

        return df.assign(**new_cols)

    #######################################
    # matches played and wins, win ratios #
    #######################################

    def find_matches_played(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates matches played so far in the player's career in total and per surface.
        Assumes that the passed dataframe is player specific.
        """
        df = df.copy()

        matches_pld = np.ones(len(df))

        # all matches
        df = df.assign(cumul_matches_played=matches_pld.cumsum())

        # per surface
        for s in self.SURFACES:
            cumul_matches_played_surface = f"cumul_matches_played_{s.lower()}"
            df = df.assign(**{cumul_matches_played_surface : (matches_pld*(df["surface"] == s)).cumsum()})

        return df

    def find_wins(self, player_id: int, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates wins so far in the player's career in total and per surface.
        """
        df = df.copy()

        win_mask = (df["winner_id"] == player_id)

        # all surfaces
        df = df.assign(cumul_wins=win_mask.cumsum())

        # per surface
        for s in self.SURFACES:
            cumul_wins_surface = f"cumul_wins_{s.lower()}"
            df = df.assign(**{cumul_wins_surface : (win_mask * (df["surface"] == s)).cumsum()})

        return df

    def find_win_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the win ratios in the player's career so far in total and per surface.
        Assumes that the passed dataframe is player specific.
        """
        df = df.copy()

        # all surfaces
        df = df.assign(cumul_win_ratio=df["cumul_wins"]/df["cumul_matches_played"])
        df["cumul_win_ratio"] = df["cumul_win_ratio"].replace([np.inf, -np.inf], np.nan).fillna(0)

        # per surface
        for s in self.SURFACES:
            cumul_win_ratio_surface = f"cumul_win_ratio_{s.lower()}"
            ratio = df[f"cumul_wins_{s.lower()}"] / df[f"cumul_matches_played_{s.lower()}"]
            df[cumul_win_ratio_surface] = ratio.replace([np.inf, -np.inf], np.nan).fillna(0)

        return df

    def find_matches_and_wins(self, player_id: int, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies all methods that are related to matches, wins and win ratios. Assumes that
        the passed dataframe is player specific.
        """
        df = df.copy()

        df = self.find_matches_played(df)
        df = self.find_wins(player_id, df)
        df = self.find_win_ratios(df)

        return df

    #############################
    # advanced ratio statistics #
    #############################

    def _calculate_advanced_ratio_statistics(self, df: pd.DataFrame, numerator_col: str, denominator_col: str, stat_prefix: str) -> pd.DataFrame:
        """
        Helper method to calculate advanced ratio statistics so far in the player's career..
        """
        df = df.copy()

        # all surfaces
        cumul_stat_name = f"cumul_{stat_prefix}"
        df[cumul_stat_name] = (df[numerator_col]/df[denominator_col]).replace([np.inf, -np.inf], np.nan).fillna(0)

        # per surface
        for s in self.SURFACES:
            cumul_stat_name_surface = f"cumul_{stat_prefix}_{s.lower()}"
            df[cumul_stat_name_surface] = \
                (df[f"{numerator_col}_{s.lower()}"]/df[f"{denominator_col}_{s.lower()}"]).replace([np.inf, -np.inf], np.nan).fillna(0)

        return df

    def find_advanced_ratio_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates advanced ratio statistics so far in the player's career.
        """
        df = df.copy()

        advanced_ratios_dict = {"1st_serve_won_ratio": ("cumul_1stWon", "cumul_1stIn"),
                                "2nd_serve_won_ratio": ("cumul_2ndWon", "cumul_2ndIn"),
                                "ace_p_point_ratio": ("cumul_ace", "cumul_svpt"),
                                "df_p_point_ratio": ("cumul_df", "cumul_svpt"),
                                "bp_saved_ratio": ("cumul_bpSaved", "cumul_bpFaced"),
                                "bp_conversion_ratio": ("cumul_bp_won", "cumul_bp_caused"),
                                "break_ratio": ("cumul_bp_won","cumul_served_games_against_player"),
                                "hold_ratio": ("cumul_serve_games_held", "cumul_SvGms"),
                                "tiebreaks_won_ratio": ("cumul_tiebreaks_won", "cumul_tiebreaks_happened"),
                                "tiebreaks_lost_ratio": ("cumul_tiebreaks_lost", "cumul_tiebreaks_happened")
                                }

        for stat, (numerator_col, denominator_col) in advanced_ratios_dict.items():
            df = self._calculate_advanced_ratio_statistics(df, numerator_col, denominator_col, stat)

        return df

    ######################
    # per match averages #
    ######################

    def _calculate_per_match_averages(self, df: pd.DataFrame, stat_prefix: str) -> pd.DataFrame:
        """
        Helper method that calculates average stats for a given player throughout their career
        and its per surface versions
        """
        df = df.copy()

        # all surfaces
        df[f"cumul_{stat_prefix}_p_match"] = (df[f"cumul_{stat_prefix}"]/df["cumul_matches_played"]).replace([np.inf, -np.inf], np.nan).fillna(0)

        # per surface
        for s in self.SURFACES:
            df[f"cumul_{stat_prefix}_p_match_{s.lower()}"] = (df[f"cumul_{stat_prefix}_{s.lower()}"]/df[f"cumul_matches_played_{s.lower()}"]).replace([np.inf, -np.inf], np.nan).fillna(0)

        return df

    def find_per_match_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates career averages for a given player for every provided statistic.
        """
        df = df.copy()

        for stat in self.STATS_ALREADY_THERE:
            df = self._calculate_per_match_averages(df, stat)

        for stat in self.STATS_EXPLICITLY_CREATED:
            df = self._calculate_per_match_averages(df, stat)

        return df

    #########################
    # cleaning and imputing #
    #########################

    def clean_player_data(self, player_id: int, df: pd.DataFrame, drop_nans: bool = False) -> pd.DataFrame:
        """
        Cleans the passed dataframe for a specific player by turning the w_stat or
        l_stat into player_stat, based on whether they won or lost the game.
        """
        if drop_nans:
            df = df.copy()
            df = df.dropna()
        else:
            df = df.copy()

        for stat in self.STATS_ALREADY_THERE:
            w_stat_col = f"w_{stat}"
            l_stat_col = f"l_{stat}"
            player_stat_col = f"player_{stat}"

            df[player_stat_col] = np.where(df["winner_id"] == player_id, df[w_stat_col], df[l_stat_col])
            df.drop(columns=[w_stat_col, l_stat_col], inplace=True)

        for stat in self.STATS_EXPLICITLY_CREATED:
            w_stat_col = f"w_{stat}"
            l_stat_col = f"l_{stat}"
            player_stat_col = f"player_{stat}"

            df[player_stat_col] = np.where(df["winner_id"] == player_id, df[w_stat_col], df[l_stat_col])
            df.drop(columns=[w_stat_col, l_stat_col], inplace=True)

        # dropping general match info and player attributes no longer needed
        df.drop(columns=self.COLS_TO_DROP_AFTER_CLEANING, inplace=True, errors='ignore')

        return df

    def impute_player_match_stats_with_lagged_avg(self, player_id: int, player_df: pd.DataFrame) -> pd.DataFrame:
        """
        Imputes missing match-specific stats (w_ace, l_ace, etc.) for a player
        using their lagged historical average for that stat.
        Assumes player_df is sorted by date and filtered for the player.
        """
        df = player_df.copy()
        is_winner = (df["winner_id"] == player_id)
        is_loser = (df["loser_id"] == player_id)

        # count of matches played by the player up to each row
        player_match_count_series = pd.Series(np.arange(1, len(df) + 1), index=df.index)

        # lagged count of matches
        lagged_player_match_count = player_match_count_series.shift(1).fillna(0)

        # Use STATS_ALREADY_THERE as these are the source columns with potential NaNs
        stats_to_impute = self.STATS_ALREADY_THERE

        for stat_prefix in stats_to_impute:
            w_col = f"w_{stat_prefix}"
            l_col = f"l_{stat_prefix}"

            if w_col not in df.columns or l_col not in df.columns:
                print(f"Warning: Columns {w_col} or {l_col} not found for imputation. Skipping {stat_prefix}.")
                continue

            temp_stat_this_match = pd.Series(0.0, index=df.index)

            # populate with winner's stats (NaNs become 0 for this avg calculation)
            winner_stats_for_avg = df.loc[is_winner, w_col].fillna(0)
            temp_stat_this_match.loc[is_winner] = winner_stats_for_avg

            # populate with loser's stats (NaNs become 0 for this avg calculation)
            loser_stats_for_avg = df.loc[is_loser, l_col].fillna(0)
            temp_stat_this_match.loc[is_loser] = loser_stats_for_avg

            # preliminary cumulative sum (based on 0-filled stats)
            prelim_cumul_stat = temp_stat_this_match.cumsum()

            # calculating lagged average
            lagged_avg_stat = (prelim_cumul_stat.shift(1).fillna(0) / lagged_player_match_count).replace([np.inf, -np.inf], np.nan).fillna(0)

            # imputing NaNs in the original w_col and l_col for the player using this lagged_avg_stat
            df.loc[is_winner & df[w_col].isna(), w_col] = lagged_avg_stat[is_winner & df[w_col].isna()]
            df.loc[is_loser & df[l_col].isna(), l_col] = lagged_avg_stat[is_loser & df[l_col].isna()]

        return df

    ######################
    # rolling statistics #
    ######################

    def rolling_sums_avgs(self, df: pd.DataFrame, recent_matches: int = 10, time_window:str = "90D"):
        """
        Method that calculates rolling sums and averages of player stats to account for recent form.
        Uses a sliding window of matches and dates.

        Assumes the passed df to be sorted by ascending date for the specified player, cleaned and imputed by previous
        methods.

        Automatically names the new features based on the recency of date and matches passed, so it can be applied
        several times if various windows are to be considered in the model.

        By default it takes into account last 10 games or last 3 months, whichever of the two is reached first.
        """
        # prelims
        df = df.copy()
        time_name = time_window
        time_window = pd.Timedelta(time_window)

        for stat in self.PLAYER_STATS:
            rolling_sums = []
            rolling_avgs = []

            window = deque()

            running_sum = 0
            running_count = 0

            # column names
            sum_col_name = f'{stat}_roll_sum_{recent_matches}_{time_name}'
            avg_col_name = f'{stat}_roll_avg_{recent_matches}_{time_name}'

            # iterator for rolling window
            for index, row in df.iterrows():
                # current row
                current_date = row["adjusted_date"]
                current_value = row[stat]

                # adding new row
                window.append((current_date, current_value))
                running_sum += current_value
                running_count += 1

                # removing rows out of the windows
                while window and ((current_date - window[0][0] > time_window) or (len(window) > recent_matches)):
                    old_date, old_value = window.popleft()
                    running_sum -= old_value
                    running_count -= 1

                # storing results
                rolling_sums.append(running_sum)
                rolling_avgs.append(running_sum/running_count if running_count > 0 else 0)

            # updating df with the new feature
            df[sum_col_name] = rolling_sums
            df[avg_col_name] = rolling_avgs

        return df

    def rolling_sums_avgs_p_surface(self, df: pd.DataFrame, recent_matches: int = 10, time_window:str = "90D"):
        """
        Calculates per-surface rolling sums and averages and adds them to the player's match history DataFrame.

        For each match in the input DataFrame, this method provides the player's recent form
        on that specific surface, as well as their last known form on other surfaces.

        Uses a sliding window of matches and dates.
        Assumes the passed df to be sorted by ascending date for the specified player.
        """
        df = df.copy()

        df_out = df.copy()
        time_name = time_window

        all_new_columns = []

        # stats for each surface and joining with original df
        for surface in self.SURFACES:

            df_surf = df[df["surface"] == surface]

            # checking if df_surf is empty, to skip this surface if True and avoid potential issues
            if df_surf.empty:
                continue

            # calculate rolling stats per surface
            df_surf_with_stats = self.rolling_sums_avgs(df_surf, recent_matches, time_window)

            new_cols = [c for c in df_surf_with_stats.columns if c.endswith(f"_{recent_matches}_{time_name}")]

            # renaming new cols per surface
            rename_dict = {col: f"{col}_{surface.lower()}" for col in new_cols}
            surface_stats_renamed = df_surf_with_stats[new_cols].rename(columns=rename_dict)

            all_new_columns.extend(surface_stats_renamed.columns)

            # joining on the index, which is preserved from the original df.
            df_out = df_out.join(surface_stats_renamed)

        if all_new_columns:

            # forward-filling new columns to propagate the last known value to subsequent matches on other surfaces
            df_out[all_new_columns] = df_out[all_new_columns].ffill()

            # filling any remaining NaNs with 0
            df_out[all_new_columns] = df_out[all_new_columns].fillna(0)

        return df_out

    def rolling_advanced_ratios(self, df: pd.DataFrame, recent_matches: int = 10, time_window_str: str = "90D"):
        """
        Method that calculates rolling ratios of advanced statistics, to account for recent form.
        It computes these by dividing the rolling sum of the numerator stat by the
        rolling sum of the denominator stat.

        Assumes the passed df is player-specific, sorted by 'adjusted_date', and that
        `rolling_sums_avgs` has already been called to create the necessary
        `player_*_roll_sum_{recent_matches}_{time_window_str}` columns.

        Args:
            df (pd.DataFrame): Player-specific DataFrame.
            recent_matches (int): The number of recent matches for the rolling window.
            time_window_str (str): The time window string (e.g., "90D", "180D", "365D").

        Returns:
            pd.DataFrame: DataFrame with added rolling advanced ratio columns.
        """
        df = df.copy()
        time_suffix = f"{recent_matches}_{time_window_str}"

        # defining numerator and denominator prefixes for the rolling sum cols
        # key: prefix for the new ratio_column (e.g., "1st_serve_won")
        # value: (numerator_sum_prefix, denominator_sum_prefix)
        # these prefixes correspond to the `player_*` stats.
        ratio_definitions = {"1st_serve_won": ("player_1stWon_roll_sum", "player_1stIn_roll_sum"),
                                "2nd_serve_won": ("player_2ndWon_roll_sum", "player_2ndIn_roll_sum"),
                                "ace_p_point": ("player_ace_roll_sum", "player_svpt_roll_sum"),
                                "df_p_point": ("player_df_roll_sum", "player_svpt_roll_sum"),
                                "bp_saved": ("player_bpSaved_roll_sum", "player_bpFaced_roll_sum"),
                                "bp_conversion": ("player_bp_won_roll_sum", "player_bp_caused_roll_sum"),
                                "break_pctg": ("player_bp_won_roll_sum","player_served_games_against_player_roll_sum"),
                                "hold_pctg": ("player_serve_games_held_roll_sum", "player_SvGms_roll_sum"),
                                "tiebreaks_won": ("player_tiebreaks_won_roll_sum", "player_tiebreaks_happened_roll_sum"),
                                "tiebreaks_lost": ("player_tiebreaks_lost_roll_sum", "player_tiebreaks_happened_roll_sum")
                                }

        for ratio_prefix, (num_prefix, den_prefix) in ratio_definitions.items():
            num_col = f"{num_prefix}_{time_suffix}"
            den_col = f"{den_prefix}_{time_suffix}"
            new_ratio_col = f"roll_{ratio_prefix}_ratio_{time_suffix}"

            if num_col in df.columns and den_col in df.columns:
                df[new_ratio_col] = (df[num_col] / df[den_col]).replace([np.inf, -np.inf], np.nan).fillna(0)
            else:
                print(f"Warning: Numerator '{num_col}' or Denominator '{den_col}' not found for '{new_ratio_col}'. Skipping.")

        return df

    def rolling_advanced_ratios_p_surface(self, df: pd.DataFrame, recent_matches: int = 10, time_window: str = "90D"):
        """
        Method that calculates advanced ratios on a per-surface basis. Note that to account for rareness of surfaces
        it (should) be applied to a wider time window, than the default 90 days.
        """
        df_out = df.copy()

        time_name = time_window

        all_new_columns = []

        # stats for each surface and joining with original df
        for surface in self.SURFACES:

            df_surf = df[df["surface"] == surface]

            # checking if df_surf is empty, to skip this surface if True and avoid potential issues
            if df_surf.empty:
                continue

            # calculate rolling stats per surface
            df_surf_with_sums = self.rolling_sums_avgs(df_surf, recent_matches, time_window)
            df_surf_with_stats = self.rolling_advanced_ratios(df_surf_with_sums, recent_matches, time_window)

            new_cols = [c for c in df_surf_with_stats.columns if c.endswith(f"roll_ratio_{recent_matches}_{time_name}")]

            # renaming new cols per surface
            rename_dict = {col: f"{col}_{surface.lower()}" for col in new_cols}
            surface_stats_renamed = df_surf_with_stats[new_cols].rename(columns=rename_dict)

            all_new_columns.extend(surface_stats_renamed.columns)

            # joining on the index, which is preserved from the original df.
            df_out = df_out.join(surface_stats_renamed)

        if all_new_columns:

            # forward-filling new columns to propagate the last known value to subsequent matches on other surfaces
            df_out[all_new_columns] = df_out[all_new_columns].ffill()

            # filling any remaining NaNs with 0
            df_out[all_new_columns] = df_out[all_new_columns].fillna(0)

        return df_out

    def generate_master_feature_dataframe(self) -> pd.DataFrame:
        """
        Generates a master DataFrame with lagged cumulative stats for all players
        across all their matches and saves it to a Parquet file.
        """

        # defining absolute path to the dir where the parquet will be
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        dir_path = os.path.join(root_dir, "processed_data", "tennis_atp", "singles")
        file_name = 'master_player_features.parquet'
        output_path = os.path.join(dir_path, file_name)

        print("Starting generation of master player-specific feature DataFrame...")

        # loading and combining all singles data from all years
        print("Loading and combining all yearly singles data...")
        all_years_dfs = []

        for year, df_year_original in self.singles_data.items():
            # appending a copy
            df_year = df_year_original.copy()
            all_years_dfs.append(df_year)

        combined_df = pd.concat(all_years_dfs, ignore_index=True)
        print(f"Combined data shape: {combined_df.shape}")

        # date offset
        print("Applying date offset globally...")
        combined_df_with_adj_date = self.date_offset(combined_df)

        # making explicit features that are needed for calculations
        print("Creating basic features...")
        combined_df_with_basic_features = self.create_basic_features(combined_df_with_adj_date)

        # sorting by adjusted date, tourney id and match_num
        print("Sorting combined data by 'adjusted_date'...")
        sort_keys = ['adjusted_date', 'tourney_id']

        if 'match_num' in combined_df_with_basic_features.columns:
            sort_keys.append('match_num')

        combined_df_with_basic_features.sort_values(by=sort_keys, inplace=True)

        # extracting player ids
        all_player_ids = sorted(list(self.set_all_player_ids(combined_df_with_basic_features)))
        print(f"Found {len(all_player_ids)} unique player IDs.")

        list_of_player_feature_dfs = []
        match_identifier_cols = ['tourney_id', 'match_num', 'adjusted_date', 'tourney_date']

        if 'match_num' not in combined_df_with_adj_date.columns:
            print("Warning: 'match_num' not in DataFrame. It will be excluded from identifiers.")
            match_identifier_cols.remove('match_num')

        # start processing
        print(f"Processing players...")
        for i, player_id in enumerate(all_player_ids):
            # showing the progress
            print(f"Processing player {i+1}/{len(all_player_ids)}: ID {player_id}")

            # per player df
            player_all_matches_df = self.find_player_rows(player_id, combined_df_with_basic_features, surface = "A")

            # imputing missing values before any calculations
            player_imputed_df = self.impute_player_match_stats_with_lagged_avg(player_id, player_all_matches_df)

            # cumulative stats
            player_cumul_stats_df = self.find_cumul_stats(player_id, player_imputed_df)

            # matches and wins
            player_matches_and_wins_df = self.find_matches_and_wins(player_id, player_cumul_stats_df)

            # advanced ratio statistics
            player_advanced_ratio_stats_df = self.find_advanced_ratio_statistics(player_matches_and_wins_df)

            # per match averages
            player_per_match_averages_df = self.find_per_match_averages(player_advanced_ratio_stats_df)

            # cleaning player data
            player_cleaned_df = self.clean_player_data(player_id, player_per_match_averages_df)

            # applying rolling stats
            player_rolling_stats_df = self.rolling_sums_avgs(player_cleaned_df, recent_matches = 10, time_window = "90D")
            player_rolling_stats_df_p_surface = self.rolling_sums_avgs_p_surface(player_rolling_stats_df, recent_matches = 10, time_window = "90D")

            player_advanced_df = self.rolling_advanced_ratios(player_rolling_stats_df_p_surface, recent_matches = 10, time_window_str = "90D")
            player_advanced_df_p_surface = self.rolling_advanced_ratios_p_surface(player_advanced_df, recent_matches = 10, time_window = "90D")

            # concantenating all player stats
            player_stats_df = player_advanced_df_p_surface.copy()

            # creating lagged stats because we want each row to have the stats up to the last match before the one in it
            lagged_player_stats_df = player_stats_df.copy()

            stat_cols = [col for col in player_stats_df.columns if col.startswith('cumul_') or '_roll_' in col]


            # creating lagged versions of stat columns of historical stats of the player
            lagged_features_df = lagged_player_stats_df[stat_cols].shift(1).fillna(0)
            lagged_features_df.columns = [f"career_{col}" for col in stat_cols]

            # combining columns, surface, and the lagged features
            final_cols_base = match_identifier_cols + ['surface']
            current_player_features_df = lagged_player_stats_df[final_cols_base].copy()

            current_player_features_df = pd.concat([current_player_features_df, lagged_features_df], axis=1)

            current_player_features_df['player_id'] = player_id
            list_of_player_feature_dfs.append(current_player_features_df)

        master_df = pd.DataFrame()

        if list_of_player_feature_dfs:
            print("Concatenating all player features...")
            master_df = pd.concat(list_of_player_feature_dfs, ignore_index=True)
            master_df.sort_values(by=['player_id', 'adjusted_date'], inplace=True)

            if output_path:
                print(f"Saving master feature DataFrame to: {output_path}")

                # ensuring directory exists

                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                master_df.to_parquet(output_path, index=False)
                print("Successfully saved to Parquet.")
        else:
            print("No player features generated. Master DataFrame is empty.")

        print(f"Master feature DataFrame generation complete. Shape: {master_df.shape}")

        return master_df
