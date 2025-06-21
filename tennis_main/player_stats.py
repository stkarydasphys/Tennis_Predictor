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

# for sliding window rolling method
from collections import deque

class Player:

    def __init__(self):
        self.singles_data = Tennis().get_singles()
        self.players_data = Tennis().get_players()

    def set_all_player_ids(self, df):
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

    #################################
    # creating 2nd serve in feature #
    #################################

    def create_2nd_serve_in(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Method that creates an 2nd serve in feature, for later ease of use.
        """
        df = df.copy()

        df.loc[:, "w_2ndIn"] = df["w_svpt"] - df["w_1stIn"] - df["w_df"]
        df.loc[:, "l_2ndIn"] = df["l_svpt"] - df["l_1stIn"] - df["l_df"]

        return df


    ########################################
    # helper function for cumulative stats #
    ########################################

    def _calculate_cumulative_stat(self, df: pd.DataFrame, player_id: int, stat_prefix: str, w_stat_col: str, l_stat_col: str) -> pd.DataFrame:
        """
        Helper method to calculate cumulative stats for a given player
        and its per-surface versions.

        For later ease, it is also used to create not only cumulative stats but their single row versions too,
        instead of creating explicit methods that do so.
        """
        surfaces = {"clay": "Clay", "grass": "Grass", "hard": "Hard", "carpet": "Carpet"}

        # filling NaNs with 0 for calculation # this is for security only, the master dataframe is created after having imputed first, see later
        df[w_stat_col] = df[w_stat_col].fillna(0)
        df[l_stat_col] = df[l_stat_col].fillna(0)

        player_stat_this_match = (df["winner_id"] == player_id) * df[w_stat_col] + (df["loser_id"] == player_id) * df[l_stat_col]

        # all surfaces
        df.loc[:, f"cumul_{stat_prefix}"] = player_stat_this_match.cumsum()

        # per surface
        for s_code, s_name in surfaces.items():
            df.loc[:, f"cumul_{stat_prefix}_{s_code}"] = (player_stat_this_match * (df["surface"] == s_name)).cumsum()

        ########################################
        # creating additional bp related feats #
        if stat_prefix == "bp_faced":
            bp_caused_on_opponent = (df["winner_id"] == player_id) * df["l_bpFaced"] + (df["loser_id"] == player_id) * df["w_bpFaced"]

            # all surfaces
            df.loc[:,"bp_caused"] = bp_caused_on_opponent
            df.loc[:,"cumul_bp_caused"] = bp_caused_on_opponent.cumsum()

            # per surface
            for surface in surfaces:
                df.loc[:, f"cumul_bp_caused_{surface}"] = \
                    (bp_caused_on_opponent * (df["surface"] == surfaces[surface])).cumsum()


        if stat_prefix == "bp_saved":
            bp_won = (df["winner_id"] == player_id) * (df["l_bpFaced"] - df["l_bpSaved"]) \
                                                    + (df["loser_id"] == player_id) * (df["w_bpFaced"]-df["w_bpSaved"])

            # all surfaces
            df.loc[:,"bp_won"] = bp_won
            df.loc[:,"cumul_bp_won"] = bp_won.cumsum()

            # per surface
            for surface in surfaces:
                df.loc[:, f"cumul_bp_won_{surface}"] = \
                    (bp_won * (df["surface"] == surfaces[surface])).cumsum()

        #######################################


        ##################################################
        # creating total games served against the player #
        if stat_prefix == "served_games":
            served_games_against_player = (df["winner_id"] == player_id) * df["l_SvGms"] + (df["loser_id"] == player_id) * df["w_SvGms"]

            # all surfaces
            df.loc[:,"served_games_against_player"] = served_games_against_player
            df.loc[:,"cumul_served_games_against_player"] = served_games_against_player.cumsum()

            # per surface
            for surface in surfaces:
                df.loc[:, f"cumul_served_games_against_player_{surface}"] = \
                    (served_games_against_player * (df["surface"] == surfaces[surface])).cumsum()

        ##################################################


        return df

    ############################
    # cumulative stats methods #
    ############################

    def cumul_aces(self, player_id, df):
        """
        Method that calculates the cumulative aces of the player. It also does so on a per
        surface basis.
        """

        return self._calculate_cumulative_stat(df, player_id, "aces", "w_ace", "l_ace")


    def cumul_df(self, player_id, df):
        """
        Method that calculates the cumulative double faults of the player. It also does so on a per
        surface basis.
        """

        return self._calculate_cumulative_stat(df, player_id, "df", "w_df", "l_df")

    def cumul_svpt(self, player_id, df):
        """
        Method that calculates the cumulative served points of the player. It also does so on a per
        surface basis.
        """

        return self._calculate_cumulative_stat(df, player_id, "svpt", "w_svpt", "l_svpt")

    def cumul_1st_serve_in(self, player_id, df):
        """
        Method that calculates the first serves of the player that were in. It also does so on a per
        surface basis.
        """

        return self._calculate_cumulative_stat(df, player_id, "1st_in", "w_1stIn", "l_1stIn")

    def cumul_2nd_serve_in(self, player_id, df):
        """
        Method that calculates the second serves of the player that were in. It also does so on a per
        surface basis.
        """

        return self._calculate_cumulative_stat(df, player_id, "2nd_in", "w_2ndIn", "l_2ndIn")


    def cumul_1st_serve_won(self, player_id, df):
        """
        Method that calculates the cumulative first serve points won. It also does so on a per
        surface basis.
        """

        return self._calculate_cumulative_stat(df, player_id, "1st_won", "w_1stWon", "l_1stWon")


    def cumul_2nd_serve_won(self, player_id, df):
        """
        Method that calculates the cumulative second serve points won. It also does so on a per
        surface basis.
        """

        return self._calculate_cumulative_stat(df, player_id, "2nd_won", "w_2ndWon", "l_2ndWon")


    def cumul_served_games(self, player_id, df):
        """
        Method that calculates the cumulative games served by the player. It also does so on a per
        surface basis.
        Moreover, it calculates the total games served against the player.
        """

        return self._calculate_cumulative_stat(df, player_id, "served_games", "w_SvGms", "l_SvGms")


    def cumul_break_points_faced(self, player_id, df):
        """
        Method that calculates the cumulative breakpoins faced. It also does so on a per
        surface basis.
        Furthermore, the helper function will calculate the cumulative breakpoints reached in favor
        of the player.
        """

        return self._calculate_cumulative_stat(df, player_id, "bp_faced", "w_bpFaced", "l_bpFaced")


    def cumul_break_points_saved(self, player_id, df):
        """
        Method that calculates the cumulative breakpoints saved. It also does so on a per
        surface basis.
        Furthermore, the helper function will calculate the cumulative breakpoints won by the player.
        """

        return self._calculate_cumulative_stat(df, player_id, "bp_saved", "w_bpSaved", "l_bpSaved")

    def cumul_matches_played(self, df) -> pd.DataFrame:
        """
        Method that calculates the total matches played. It also does so on a per surface basis.
        """
        # matches played
        matches_pld = np.ones(len(df))

        # all matches
        df.loc[:, "cumul_matches_played"] = matches_pld.cumsum()

        # per surface
        df.loc[:, "cumul_matches_played_clay"] = (matches_pld*(df["surface"] == "Clay")).cumsum()

        df.loc[:, "cumul_matches_played_grass"] = (matches_pld*(df["surface"] == "Grass")).cumsum()

        df.loc[:, "cumul_matches_played_hard"] = (matches_pld*(df["surface"] == "Hard")).cumsum()

        df.loc[:, "cumul_matches_played_carpet"] = (matches_pld*(df["surface"] == "Carpet")).cumsum()

        return df.copy()

    def cumul_wins(self, player_id: int, df: pd.DataFrame) -> pd.DataFrame:
        """
        Method that calculates the total wins so far in the player's career and also on a per
        surface basis. It also calculates the wins per match played ratio
        """

        df = df.copy()

        win_mask = (df["winner_id"] == player_id)

        surfaces = {"clay": "Clay", "grass": "Grass", "hard": "Hard", "carpet": "Carpet"}  # new


        # all surfaces wins
        df.loc[:, "cumul_wins"] = win_mask.cumsum()

        for s_code, s_name in surfaces.items():
            surface_mask = (df["surface"] == s_name)
            df.loc[:, f"cumul_wins_{s_code}"] = (win_mask * surface_mask).cumsum()

        return df

    def cumul_hold_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Method that calculates hold statistics for the player. Includes:
        - Cumulative games held
        - Hold percentage: the percentage of service games won
        """
        df = df.copy()
        # all surfaces
        df.loc[:, "cumul_games_held"] = df["cumul_served_games"] - (df["cumul_bp_faced"] - df["cumul_bp_saved"])

        # per surface
        surfaces = ["clay", "grass", "hard", "carpet"]

        for surface in surfaces:
            df.loc[:, f"cumul_games_held_{surface}"] = \
                df[f"cumul_served_games_{surface}"] - (df[f"cumul_bp_faced_{surface}"] - df[f"cumul_bp_saved_{surface}"])

        return df

    def apply_all_cumulative_for_player(self, player_id: int, player_specific_sorted_df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies all cumulative stat methods to a DataFrame already filtered for a specific player
        and sorted by 'adjusted_date'.
        The input df contains all matches for a single player, sorted by 'adjusted_date'.
        """
        new_df = player_specific_sorted_df.copy()

        # creating check for 2ndIn so that there is no conflict between the master method and the method used on its own
        if "w_2ndIn" not in new_df.columns or "l_2ndIn" not in new_df.columns:
            new_df = self.create_2nd_serve_in(new_df)

        # applying all cumulative methods
        new_df = self.cumul_aces(player_id, new_df)
        new_df = self.cumul_1st_serve_in(player_id, new_df)
        new_df = self.cumul_df(player_id, new_df)
        new_df = self.cumul_svpt(player_id, new_df)
        new_df = self.cumul_2nd_serve_in(player_id, new_df)
        new_df = self.cumul_1st_serve_won(player_id, new_df)
        new_df = self.cumul_2nd_serve_won(player_id, new_df)
        new_df = self.cumul_break_points_faced(player_id, new_df)
        new_df = self.cumul_break_points_saved(player_id, new_df)
        new_df = self.cumul_served_games(player_id, new_df)
        new_df = self.cumul_matches_played(new_df)
        new_df = self.cumul_wins(player_id, new_df)
        new_df = self.cumul_hold_stats(new_df)

        return new_df


    ############################################
    # helper function for advanced ratio stats #
    ############################################

    def _advanced_ratio_stats(self, df: pd.DataFrame, numerator_stat: str, denominator_stat: str, stat_prefix: str) -> pd.DataFrame:
        """
        Method that calculates advanced ratio statistics on total and a per
        surface basis.
        """
        df = df.copy()

        surfaces = ["clay", "grass", "hard", "carpet"]

        # all surfaces
        df.loc[:, f"cumul_{stat_prefix}_ratio"] = (df[numerator_stat]/df[denominator_stat]).replace([np.inf, -np.inf], np.nan).fillna(0)

        # per surface
        for surface in surfaces:
            df.loc[:, f"cumul_{stat_prefix}_ratio_{surface}"] = \
                (df[f"{numerator_stat}_{surface}"]/df[f"{denominator_stat}_{surface}"]).replace([np.inf, -np.inf], np.nan).fillna(0)

        return df

    ################################
    # advanced ratio stats methods #
    ################################

    def cumul_1st_serve_won_ratio(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Method that calculates the 1st serves that were won by the server. It also does so on a per
        surface basis.
        """

        return self._advanced_ratio_stats(df, "cumul_1st_won", "cumul_1st_in", "1st_serve_won")

    def cumul_2nd_serve_won_ratio(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Method that calculates the 2nd serves that were won by the server. It also does so on a per
        surface basis.
        """

        return self._advanced_ratio_stats(df, "cumul_2nd_won", "cumul_2nd_in", "2nd_serve_won")

    def cumul_bp_saved_ratio(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Method that calculates the breakpoints saved by the server. It also does so on a per
        surface basis.
        """

        return self._advanced_ratio_stats(df, "cumul_bp_saved", "cumul_bp_faced", "bp_saved")

    def cumul_aces_p_point_ratio(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Method that calculates the aces per point ratio. It also does so on a per
        surface basis.
        """

        return self._advanced_ratio_stats(df, "cumul_aces", "cumul_svpt", "aces_p_point")

    def cumul_bp_conversion_rate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Method that calculates the breakpoints conversion rate. It also does so on a per
        surface basis.
        """

        return self._advanced_ratio_stats(df, "cumul_bp_won", "cumul_bp_caused", "bp_conversion_rate")

    def cumul_break_pctg(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Method that calculates the break percentage. It also does so on a per
        surface basis.
        """

        return self._advanced_ratio_stats(df, "cumul_bp_won", "cumul_served_games_against_player", "break_pctg")

    def cumul_hold_percentage(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Method that calculates the hold percentage. It also does so on a per
        surface basis.
        """

        return self._advanced_ratio_stats(df, "cumul_games_held", "cumul_served_games", "hold_percentage")

    def cumul_win_ratio(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Method that calculates the win ratio. It also does so on a per
        surface basis.
        """

        return self._advanced_ratio_stats(df, "cumul_wins", "cumul_matches_played", "win_ratio")

    def apply_all_advanced_ratio_for_player(self, player_specific_cumulated_df: pd.DataFrame) -> pd.DataFrame:

        """
        Method that applies all advanced ratio method on a player specific dataframe.
        """
        new_df = player_specific_cumulated_df.copy()

        # applying all average methods
        new_df  = self.cumul_1st_serve_won_ratio(new_df)
        new_df  = self.cumul_2nd_serve_won_ratio(new_df)
        new_df  = self.cumul_bp_saved_ratio(new_df)
        new_df  = self.cumul_aces_p_point_ratio(new_df)
        new_df  = self.cumul_bp_conversion_rate(new_df)
        new_df  = self.cumul_break_pctg(new_df)
        new_df  = self.cumul_hold_percentage(new_df)
        new_df  = self.cumul_win_ratio(new_df)

        return new_df


    ###############################################
    # helper function for per match average stats #
    ###############################################

    def _calculate_average_p_match_stat(self, df: pd.DataFrame, stat_prefix: str) -> pd.DataFrame:
        """
        Helper method to calculate average stat for a given player *throughout* their career,
        and its per-surface versions.

        Rolling averages for recent results are calculated by different methods.
        """

        # all surfaces average stat
        df.loc[:, f'{stat_prefix}_p_match'] = df[f'cumul_{stat_prefix}']/df["cumul_matches_played"]

        # per surface average stat

        surfaces = ["clay", "grass", "hard", "carpet"]

        for surface in surfaces:
            df.loc[:, f'{stat_prefix}_p_match_{surface}'] = \
                (df[f'cumul_{stat_prefix}_{surface}']/df[f'cumul_matches_played_{surface}']).fillna(0)

        return df


    ###################################
    # per match average stats methods #
    ###################################

    def aces_p_match(self, df):
        """
        Method that finds the aces per game of the player throughout their career so far.
        It also does so on a per surface basis.
        """
        return self._calculate_average_p_match_stat(df, "aces")

    def df_p_match(self, df):
        """
        Method that finds the double faults per match of the player throughout their career so far.
        It also does so on a per surface basis.
        """
        return self._calculate_average_p_match_stat(df, "df")

    def svpt_p_match(self, df):
        """
        Method that finds the served points per match of the player throughout their career so far.
        It also does so on a per surface basis.
        """
        return self._calculate_average_p_match_stat(df, "svpt")

    def first_serve_in_p_match(self, df):
        """
        Method that finds the 1st serves in per match of the player throughout their career so far.
        It also does so on a per surface basis.
        """
        return self._calculate_average_p_match_stat(df, "1st_in")

    def first_serve_won_p_match(self, df):
        """
        Method that finds the first serves won per match of the player throughout their career so far.
        It also does so on a per surface basis.
        """
        return self._calculate_average_p_match_stat(df, "1st_won")

    def second_serve_won_p_match(self, df):
        """
        Method that finds the second serves won per match of the player throughout their career so far.
        It also does so on a per surface basis.
        """
        return self._calculate_average_p_match_stat(df, "2nd_won")

    def games_served_p_match(self, df):
        """
        Method that finds the games served per match of the player throughout their career so far.
        It also does so on a per surface basis.
        """
        return self._calculate_average_p_match_stat(df, "served_games")

    def bp_faced_p_match(self, df):
        """
        Method that finds the breakpoints sfaced per match of the player throughout their career so far.
        It also does so on a per surface basis.
        """
        return self._calculate_average_p_match_stat(df, "bp_faced")

    def bp_saved_p_match(self, df):
        """
        Method that finds the breakpoints saved per match of the player throughout their career so far.
        It also does so on a per surface basis.
        """
        return self._calculate_average_p_match_stat(df, "bp_saved")

    def apply_all_average_for_player(self, player_specific_cumulated_df: pd.DataFrame) -> pd.DataFrame:

        """
        Method that applies all averaging methods. Assumes the passed df is player specific,
        sorted by 'adjusted_date'. Returns a player specific dataframe that has all the cumulative
        and average stats throughout the player's career.

        Rolling averages for recent results are calculated by different methods.
        """
        new_df = player_specific_cumulated_df.copy()

        # applying all average methods
        new_df = self.aces_p_match(new_df)
        new_df = self.df_p_match(new_df)
        new_df = self.svpt_p_match(new_df)
        new_df = self.first_serve_in_p_match(new_df)
        new_df = self.first_serve_won_p_match(new_df)
        new_df = self.second_serve_won_p_match(new_df)
        new_df = self.games_served_p_match(new_df)
        new_df = self.bp_faced_p_match(new_df)
        new_df = self.bp_saved_p_match(new_df)

        return new_df

    ##############################
    # cleaning method per player #
    ##############################

    def clean_player(self, player_id:int, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the passed dataframe for a specific player, by turning the w_stat or l_stat
        into player_stat, based on whether they won or lost that match.
        """
        df = df.copy()
        # stat names
        stat_names = ['ace', 'df', 'svpt', '1stIn', '2ndIn', '1stWon', '2ndWon', 'SvGms', 'bpSaved', 'bpFaced']

        # creating new column for each stat, specific to the player
        for stat in stat_names:
            # ensuring columns exist, especially for 2ndIn which is derived
            w_stat_col_check = f'w_{stat}'
            l_stat_col_check = f'l_{stat}'

            if w_stat_col_check not in df.columns or l_stat_col_check not in df.columns:
                print(f"Warning: Source columns {w_stat_col_check} or {l_stat_col_check} not found in clean_player. Skipping match_{stat}.")
                continue

            winner_stat_col = f'w_{stat}'
            loser_stat_col = f'l_{stat}'
            match_stat_col = f'match_{stat}'

            df[match_stat_col] = np.where(df["winner_id"] == player_id, df[winner_stat_col], df[loser_stat_col])

            df.drop(columns = [loser_stat_col, winner_stat_col], inplace = True)

        df.drop(columns = ['winner_id', 'winner_seed', 'winner_entry', 'winner_name', 'winner_hand', 'winner_ht',
                           'winner_ioc', 'winner_age', 'loser_id', 'loser_seed', 'loser_entry', 'loser_name',
                           'loser_hand', 'loser_ht', 'loser_ioc', 'loser_age', 'score', 'best_of', 'round',
                           'minutes', 'winner_rank', 'winner_rank_points', 'loser_rank', 'loser_rank_points',
                           'tourney_name', 'tourney_level', 'draw_size'], inplace = True)

        # handling the rest per-match stats created by _calculate_cumulative_stat
        # these are already player-specific for the match by the way they were calculated earlier

        if 'bp_caused' in df.columns:
            df.rename(columns={'bp_caused': 'match_bp_caused'}, inplace=True)

        if 'bp_won' in df.columns:
            df.rename(columns={'bp_won': 'match_bp_won'}, inplace=True)

        if 'served_games_against_player' in df.columns:
            df.rename(columns={'served_games_against_player': 'match_served_games_against_player'}, inplace=True)

        df.loc[:, 'match_games_held'] = df['match_SvGms'] - (df['match_bpFaced'] - df['match_bpSaved'])

        return df


   ###############################################################
   # imputing player stats with their average instead of plain 0 #
   ###############################################################

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

        stats_to_impute = [
            ("aces", "w_ace", "l_ace"),
            ("df", "w_df", "l_df"),
            ("svpt", "w_svpt", "l_svpt"),
            ("1st_in", "w_1stIn", "l_1stIn"),
            ("2nd_in", "w_2ndIn", "l_2ndIn"),
            ("1st_won", "w_1stWon", "l_1stWon"),
            ("2nd_won", "w_2ndWon", "l_2ndWon"),
            ("served_games", "w_SvGms", "l_SvGms"),
            ("bp_faced", "w_bpFaced", "l_bpFaced"),
            ("bp_saved", "w_bpSaved", "l_bpSaved"),
        ]

        for stat_prefix, w_col, l_col in stats_to_impute:
            # creating a temporary series of the player's stat for each match.
            # to calculate the preliminary average, temporarily fill NaNs in source w_col/l_col with 0.
            if w_col not in df.columns or l_col not in df.columns:
                print(f"Warning: Columns {w_col} or {l_col} not found for imputation. Skipping {stat_prefix}.")
                continue

            temp_stat_this_match = pd.Series(0.0, index=df.index)

            # populate with winner's stats (NaNs become 0 for this)
            if w_col in df.columns: # Ensure column exists
                winner_stats_for_avg = df.loc[is_winner, w_col].fillna(0)
                temp_stat_this_match.loc[is_winner] = winner_stats_for_avg

            # populate with loser's stats (NaNs become 0 for this)
            if l_col in df.columns: # Ensure column exists
                loser_stats_for_avg = df.loc[is_loser, l_col].fillna(0)
                temp_stat_this_match.loc[is_loser] = loser_stats_for_avg

            # preliminary cumulative sum (based on 0-filled stats)
            prelim_cumul_stat = temp_stat_this_match.cumsum()

            # calculating lagged average: (cumulative sum up to previous match) / (matches played up to previous match)
            # .shift(1) gets the value from the previous row. fillna(0) for the first row or if division by zero.
            lagged_avg_stat = (prelim_cumul_stat.shift(1).fillna(0) / lagged_player_match_count).fillna(0)

            # imputing NaNs in the original w_col and l_col for the player using this lagged_avg_stat
            if w_col in df.columns:
                df.loc[is_winner & df[w_col].isna(), w_col] = lagged_avg_stat[is_winner & df[w_col].isna()]
            if l_col in df.columns:
                df.loc[is_loser & df[l_col].isna(), l_col] = lagged_avg_stat[is_loser & df[l_col].isna()]

        return df

    ###########################
    # rolling average methods #
    ###########################

    def rolling_sums_avgs(self, df: pd.DataFrame, recent_matches: int = 10, time_window:str = "90D"):
        """
        Method that calculates rolling sums and averages of player stats to account for recent form.
        Uses a sliding window of matches and dates.

        Assumes the passed df to be sorted by ascending date for the specified player.
        Automatically names the new features based on the recency of date and matches passed, so it can be applied
        several times if various windows are to be considered in the model.

        By default it takes into account last 10 games or last 3 months, whichever of the two is reached first.
        """
        # prelims
        df = df.copy()
        time_name = time_window
        time_window = pd.Timedelta(time_window)

        stat_cols = ['match_ace', 'match_df', 'match_svpt', 'match_1stIn', 'match_2ndIn', 'match_1stWon',
                    'match_2ndWon', 'match_SvGms', 'match_bpSaved', 'match_bpFaced',
                    'match_bp_caused', 'match_bp_won', 'match_served_games_against_player', 'match_games_held']

        # iterator for statistic worked on currently
        for stat in stat_cols:

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

    def rolling_advanced_ratios(self, df: pd.DataFrame, recent_matches: int = 10, time_window_str: str = "90D"):
        """
        Method that calculates rolling ratios of advanced statistics, to account for recent form.
        It computes these by dividing the rolling sum of the numerator stat by the
        rolling sum of the denominator stat.

        Assumes the passed df is player-specific, sorted by 'adjusted_date', and that
        `rolling_sums_avgs` has already been called to create the necessary
        `match_*_roll_sum_{recent_matches}_{time_window_str}` columns.

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
        # T
        # these prefixes correspond to the `match_*` stats.
        ratio_definitions = {
            "1st_serve_won": ("match_1st_won_roll_sum", "match_1st_in_roll_sum"),
            "2nd_serve_won": ("match_2nd_won_roll_sum", "match_2nd_in_roll_sum"),
            "bp_saved": ("match_bp_saved_roll_sum", "match_bp_faced_roll_sum"),
            "aces_p_point": ("match_ace_roll_sum", "match_svpt_roll_sum"),
            "bp_conversion": ("match_bp_won_roll_sum", "match_bp_caused_roll_sum"),
            "break_pctg_vs_opp": ("match_bp_won_roll_sum", "match_served_games_against_player_roll_sum"),
            "hold_pctg": ("match_games_held_roll_sum", "match_SvGms_roll_sum")
        }

        surfaces = ["clay", "grass", "hard", "carpet"]

        for ratio_prefix, (num_prefix, den_prefix) in ratio_definitions.items():
            # all surfaces
            num_col = f"{num_prefix}_{time_suffix}"
            den_col = f"{den_prefix}_{time_suffix}"
            new_ratio_col = f"roll_{ratio_prefix}_ratio_{time_suffix}"

            if num_col in df.columns and den_col in df.columns:
                df[new_ratio_col] = (df[num_col] / df[den_col]).replace([np.inf, -np.inf], np.nan).fillna(0)
            else:
                print(f"Warning: Numerator '{num_col}' or Denominator '{den_col}' not found for '{new_ratio_col}'. Skipping.")

        #     # per surface
        #     for surface in surfaces:
        #         num_col_s = f"{num_prefix}_{surface}_{time_suffix}" # This assumes rolling_sums_avgs creates per-surface sums
        #         den_col_s = f"{den_prefix}_{surface}_{time_suffix}" # e.g. match_1st_won_roll_sum_clay_10_90D
        #         new_ratio_col_s = f"roll_{ratio_prefix}_ratio_{surface}_{time_suffix}"

        #         # This part needs adjustment: rolling_sums_avgs currently doesn't create per-surface rolling sums.
        #         # If you need per-surface rolling ratios, rolling_sums_avgs would need to be enhanced
        #         # or this method would need to perform rolling sums on surface-filtered data first.
        #         # For now, let's assume rolling_sums_avgs is enhanced or we skip per-surface here.
        #         # To implement per-surface rolling ratios correctly here, you'd first filter df_copy by surface,
        #         # then call a modified rolling_sums_avgs (or a helper) on that subset for num_prefix and den_prefix,
        #         # then calculate the ratio, and then merge/assign back to df_copy.
        #         # This is complex. A simpler start is to ensure rolling_sums_avgs can produce
        #         # match_STAT_clay_roll_sum_WINDOW etc.
        #         # However, your current rolling_sums_avgs iterates through a fixed list of pl_stats
        #         # and doesn't have a per-surface mechanism within its loop.

        #         # --- Simplified approach: If rolling_sums_avgs does NOT produce per-surface sums ---
        #         # You would need to calculate these ratios based on career cumulative per-surface stats,
        #         # or implement per-surface rolling sums in rolling_sums_avgs.
        #         # The current structure of rolling_sums_avgs is not set up for per-surface easily.
        #         # Let's assume for now we are focusing on the "all surfaces" rolling ratios.
        #         # If per-surface rolling sums (e.g., match_ace_roll_sum_clay_10_90D) are indeed created by
        #         # an enhanced rolling_sums_avgs, then the following would work:

        #         if num_col_s in df_copy.columns and den_col_s in df_copy.columns:
        #              df_copy[new_ratio_col_s] = (df_copy[num_col_s] / df_copy[den_col_s]).replace([np.inf, -np.inf], np.nan).fillna(0)
        #         # else:
        #         #     print(f"Warning: Numerator '{num_col_s}' or Denominator '{den_col_s}' not found for '{new_ratio_col_s}'. Skipping surface.")


        # # Win Ratio (Matches Won / Matches Played) - Rolling
        # # Requires rolling sum of wins and rolling count of matches played.
        # # 'rolling_sums_avgs' would need to process a 'match_won_indicator' (1 if player won, 0 if lost)
        # # and a 'match_played_indicator' (always 1 for matches player participated in).

        # # Example for rolling win ratio (requires 'match_won_indicator_roll_sum' and 'match_played_indicator_roll_sum')
        # # if f"match_won_indicator_roll_sum{time_suffix}" in df_copy.columns and \
        # #    f"match_played_indicator_roll_sum{time_suffix}" in df_copy.columns:
        # #     df_copy[f"roll_win_ratio{time_suffix}"] = \
        # #         (df_copy[f"match_won_indicator_roll_sum{time_suffix}"] /
        # #          df_copy[f"match_played_indicator_roll_sum{time_suffix}"]).replace([np.inf, -np.inf], np.nan).fillna(0)

        # #     for surface in surfaces:
        # #         if f"match_won_indicator_roll_sum_{surface}{time_suffix}" in df_copy.columns and \
        # #            f"match_played_indicator_roll_sum_{surface}{time_suffix}" in df_copy.columns:
        # #             df_copy[f"roll_win_ratio_{surface}{time_suffix}"] = \
        # #                 (df_copy[f"match_won_indicator_roll_sum_{surface}{time_suffix}"] /
        # #                  df_copy[f"match_played_indicator_roll_sum_{surface}{time_suffix}"]).replace([np.inf, -np.inf], np.nan).fillna(0)
        # # else:
        # #     print(f"Warning: Columns for rolling win ratio not found. Skipping.")


        return df



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

        # date_offset for all to create 'adjusted_date'
        print("Applying date offset globally...")
        combined_df_with_adj_date = self.date_offset(combined_df) # date_offset handles copying

        # sorting dataset by 'adjusted_date' and other keys for stability
        print("Sorting combined data by 'adjusted_date'...")
        sort_keys = ['adjusted_date', 'tourney_id']

        if 'match_num' in combined_df_with_adj_date.columns:
            sort_keys.append('match_num')

        combined_df_with_adj_date.sort_values(by=sort_keys, inplace=True)

        # all unique player IDs
        all_player_ids = sorted(list(self.set_all_player_ids(combined_df_with_adj_date)))
        print(f"Found {len(all_player_ids)} unique player IDs.")

        # processing each player
        list_of_player_feature_dfs = []
        match_identifier_cols = ['tourney_id', 'match_num', 'adjusted_date', 'tourney_date']

        # ensuring 'match_num' is in the df, or handling its absence
        if 'match_num' not in combined_df_with_adj_date.columns:
            print("Warning: 'match_num' not in DataFrame. It will be excluded from identifiers.")
            match_identifier_cols.remove('match_num')

        # start processing
        print(f"Processing players...")
        for i, player_id in enumerate(all_player_ids):
            # showing the progress to the user
            print(f"Processing player {i+1}/{len(all_player_ids)}: ID {player_id}")

            # per player df
            player_all_matches_df = self.find_player_rows(player_id, combined_df_with_adj_date, surface = "A")

            # creating extra features
            player_all_matches_df_extra = self.create_2nd_serve_in(player_all_matches_df)

            # imputing with player's lagged historical average if there existed NaNs in the stats
            player_all_matches_df = self.impute_player_match_stats_with_lagged_avg(player_id, player_all_matches_df_extra)

            # per player cumulative stats, average stats and rolling stats
            player_cumulative_stats_df = self.apply_all_cumulative_for_player(player_id, player_all_matches_df)
            player_average_stats_df = self.apply_all_average_for_player(player_cumulative_stats_df)
            player_advanced_ratio_stats_df = self.apply_all_advanced_ratio_for_player(player_average_stats_df)
            player_clean_stats_df = self.clean_player(player_id, player_advanced_ratio_stats_df)
            player_rolling_stats_df = self.rolling_sums_avgs(player_clean_stats_df)

            # concantenating all player stats
            player_stats_df = player_rolling_stats_df.copy()

            # creating lagged stats because we want each row to have the stats up to the last match before the one in it
            lagged_player_stats_df = player_stats_df.copy()

            stat_cols = [col for col in player_stats_df.columns if col.startswith('cumul_') or \
                col.endswith('_p_match') or '_roll_' in col or col.startswith('wins_') or col.startswith('win_ratio_') \
                    or col.endswith('_clay') or col.endswith('_grass') or col.endswith('_hard') or col.endswith('_carpet')]

            # creating lagged versions of stat columns of historical stats of the player
            lagged_features_df = lagged_player_stats_df[stat_cols].shift(1).fillna(0)   ####
            lagged_features_df.columns = [f"career_{col}" for col in stat_cols]

            # combining columns, surface, and the lagged features
            final_cols_base = match_identifier_cols + ['surface']
            current_player_features_df = lagged_player_stats_df[final_cols_base].copy()   ####

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
