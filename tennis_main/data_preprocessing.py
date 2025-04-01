"""
As of April 1st, this script includes all the preprocessing steps for the singles
data we did, for the MVP. We handle duplicates if they exist, drop all match
statistics (which later will be integrated in player related .csv files), and
deal with missing values and with player seed when missing, while also creating
binary column for the players that were seeded or not.)

After that, winner and loser columns are renamed and randomized, and a target
column is created so that it shows whether player A or player B won.

Then specific features are picked, and fed to a scaler and an encoder. The
player seed is dropped for now and the encoded and scaled features recombined.
"""
# general imports
import pandas as pd
import numpy as np

# project related imports
from tennis_main.data import Tennis

# model related
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


class Tennis_preprocessing:
    def __init__(self):
        self.singles_data =Tennis.get_singles()
        self.players_data = Tennis.get_players()
        self.rankings_data = Tennis.get_rankings()

    def duplicates_and_match_stats(self, df):
        """
        Method that drops duplicate entries when called on a dataframe that
        contains singles' data. It also drops match statistics that are not
        taken into account for the MVP model.
        """

        # duplicates
        df = df.drop_duplicates(inplace = True)

        # match statistics
        stats_columns = ['minutes', 'w_ace', 'w_df', 'w_svpt', 'w_1stIn',
                         'w_1stWon', 'w_2ndWon', 'w_SvGms', 'w_bpSaved',
                         'w_bpFaced', 'l_ace', 'l_df', 'l_svpt',
                        'l_1stIn', 'l_1stWon', 'l_2ndWon', 'l_SvGms',
                        'l_bpSaved', 'l_bpFaced']
        df = df.drop(stats_columns, axis = 1)

        return df

    def missing_values(self, df):
        """
        Method that deals with missing values.
        In columns winner_ht, loser_ht the missing values are imputed with
        the mean value.

        In columns winner_entry, loser_entry the missing values are imputed
        with 'R'.

        In winner_rank and loser_rank we impute a rank that is greater than the
        greatest rand in the dataframe, since these players are probably not
        ranked at all. Their rank points are filled with 0s.

        Finally, player_seed when NaN is filled with the player's rank. At the same
        time, we create a binary column for when the player was seeded or not.
        """

        df['winner_ht'] = df['winner_ht'].fillna(df['winner_ht'].mean())
        df['loser_ht'] = df['loser_ht'].fillna(df['loser_ht'].mean())

        df['winner_entry'] = df['winner_entry'].fillna('R')
        df['loser_entry'] = df['loser_entry'].fillna('R')

        df['winner_rank'] = df['winner_rank'].fillna(df['winner_rank'].max() + 1)
        df['loser_rank'] = df['loser_rank'].fillna(df['loser_rank'].max() + 1)

        df['winner_rank_points'] = df['winner_rank_points'].fillna(0)
        df['loser_rank_points'] = df['loser_rank_points'].fillna(0)

        df['winner_seed'] = df['winner_seed'].fillna(df['winner_rank'])
        df['loser_seed'] = df['loser_seed'].fillna(df['loser_rank'])

        df['winner_is_seeded'].apply(lambda x: 1 if pd.notna(x) else 0)
        df['loser_is_seeded'].apply(lambda x: 1 if pd.notna(x) else 0)

        return df

    def target_column(self, df):
        """
        Method that turns winner and loser into player_A and player_B.
        Then, randomly interchanges their values and creates a target column
        that contains 0 or 1, depending on who won (player A or player B).

        Finally, drops all the features that will not be used.
        """

        # renaming

        df["original_winner_id"] = df["winner_id"].copy()

        renamed_columns = {
            'winner_id': 'player_A_id',
            'winner_seed': 'player_A_seed',
            'winner_entry': 'player_A_entry',
            'winner_name': 'player_A_name',
            'winner_hand': 'player_A_hand',
            'winner_ht': 'player_A_ht',
            'winner_ioc': 'player_A_ioc',
            'winner_age': 'player_A_age',
            'winner_rank': 'player_A_rank',
            'winner_rank_points': 'player_A_rank_points',
            'winner_is_seeded': 'player_A_is_seeded',
            'loser_id': 'player_B_id',
            'loser_seed': 'player_B_seed',
            'loser_entry': 'player_B_entry',
            'loser_name': 'player_B_name',
            'loser_hand': 'player_B_hand',
            'loser_ht': 'player_B_ht',
            'loser_ioc': 'player_B_ioc',
            'loser_age': 'player_B_age',
            'loser_rank': 'player_B_rank',
            'loser_rank_points': 'player_B_rank_points',
            'loser_is_seeded': 'player_B_is_seeded',
        }

        df.rename(columns=renamed_columns, inplace=True)

        # swapping
        columns_to_swap = [
            'player_A_id', 'player_B_id',
            'player_A_seed', 'player_B_seed',
            'player_A_entry', 'player_B_entry',
            'player_A_name', 'player_B_name',
            'player_A_hand', 'player_B_hand',
            'player_A_ht', 'player_B_ht',
            'player_A_ioc', 'player_B_ioc',
            'player_A_age', 'player_B_age',
            'player_A_rank', 'player_B_rank',
            'player_A_rank_points', 'player_B_rank_points',
            'player_A_is_seeded', 'player_B_is_seeded',
        ]

        np.random.seed(2010)
        swap_mask = np.random.choice([True, False], size=len(df))

        for col_A, col_B in zip(columns_to_swap[::2], columns_to_swap[1::2]):
            df.loc[swap_mask, [col_A, col_B]] = df.loc[swap_mask, [col_B, col_A]].values

        # creating target column (0 if player A won, 1 if player B won)
        df["target"] = 0  # initialize all to 0 (player A won)
        df.loc[df["player_A_id"] != df["original_winner_id"], "target"] = 1  # change to 1 if player B won

        # dropping columns that are not to be used in the MVP
        drop_cols = ["tourney_name", "tourney_date", "player_A_ioc", "player_B_ioc",
                     "score", "winner_id", "player_A_name", "player_B_name",
                    "player_A_id", "player_B_id", "tourney_id", "draw_size",
                    "match_num", "player_A_ht", "player_B_ht", "round", "original_winner_id"]

        df.drop(columns=drop_cols, inplace=True, errors='ignore')

        return df

    def clean_singles_MVP(self):
        """
        This method applies all the cleaning methods created so far to each
        of the keys in the self.data dictionary.

        Keep in mind that all these methods are based on the MVP methodology
        and subsequently my need tweaking.
        """

        for key in self.singles_data.keys():
            self.singles_data[key] = self.duplicates_and_match_stats(self.singles_data[key])
            self.singles_data[key] = self.missing_values(self.singles_data[key])
            self.singles_data[key] = self.target_column(self.singles_data[key])

        return self.singles_data

    def create_X_and_y(self,df):
        """
        Separates the df into features and target
        """

        cols = list(df.columns)
        cols.remove('target')
        X = df[cols]
        y = df['target']

        return X, y

    def split_and_scale(self, X, y, test_size = 0.3, random_state = 2010):
        """
        Performs Train/Test split at a ratio of 0.7:0.3 unless otherwise specified.
        Starts with a specific seed unless otherwise specified.

        Then, applies MinMaxScaling to the train set only, but transforms both sets
        and returns their scaled versions.
        """
        # split
        X_train, X_test, y_train, y_test = train_test_split(X, y, \
                test_size = test_size, random_state = random_state)

        # scaling
        feats_to_scale = ["player_A_seed", "player_A_age", "player_B_seed",
                          "player_B_age", "player_A_rank", "player_A_rank_points",
                          "player_B_rank", "player_B_rank_points"]

        scaler = MinMaxScaler()

        # fitting to train, transforming both
        scaler.fit(X_train[feats_to_scale])
        X_train[feats_to_scale] = scaler.transform(X_train[feats_to_scale])
        X_test[feats_to_scale] = scaler.transform(X_test[feats_to_scale])

        # resetting index of targets for later use
        y_train = y_train.reset_index().drop(columns = "index")
        y_test = y_test.reset_index().drop(columns = "index")

        return X_train, X_test, y_train, y_test

    def encode(self, X_train, X_test):
        """
        Performs encoding in appropriate features. OneHotEncoder is fitted on train
        but applied on both train and test to transform.

        Returns X_train and X_test with encoded features. These need recombining
        with the scaled numerical features that is performed by another function.
        """

        feats_to_encode = ["surface", "tourney_level", "player_A_entry",
                           "player_B_entry", "player_A_hand", "player_B_hand"]

        # instantiating encoder
        # we drop the first category for each feature encoded, to reduce complexity
        ohe = OneHotEncoder(sparse_output = False, drop = "first")

        # fitting to train set ONLY!
        ohe.fit(X_train[feats_to_encode])

        # transforming
        X_train_encoded = ohe.transform(X_train[feats_to_encode])
        X_test_encoded = ohe.transform(X_test[feats_to_encode])

        # converting into dataframes
        X_train_encoded_df = pd.DataFrame(X_train_encoded, columns=ohe.get_feature_names_out())
        X_test_encoded_df = pd.DataFrame(X_test_encoded, columns=ohe.get_feature_names_out())

        # resetting indices
        X_train.reset_index(inplace = True)
        X_test.reset_index(inplace = True)

        return X_train, X_test
