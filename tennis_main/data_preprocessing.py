"""
As of April 11th, this script includes all the preprocessing steps for the singles
data, for the MVP model. We handle duplicates if they exist, drop all match
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

# data preprocessing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder

# model evaluation
from sklearn.model_selection import train_test_split

# module constants
SEED = 2010

STATS_COLUMNS_TO_DROP = ['minutes', 'w_ace', 'w_df', 'w_svpt', 'w_1stIn',
                         'w_1stWon', 'w_2ndWon', 'w_SvGms', 'w_bpSaved',
                         'w_bpFaced', 'l_ace', 'l_df', 'l_svpt',
                        'l_1stIn', 'l_1stWon', 'l_2ndWon', 'l_SvGms',
                        'l_bpSaved', 'l_bpFaced']

RENAMED_COLUMNS = {
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

COLUMNS_TO_RANDOM_SWAP = [
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

MVP_COLUMNS_TO_DROP = ["tourney_name", "tourney_date", "player_A_ioc", "player_B_ioc",
                     "score", "winner_id", "player_A_name", "player_B_name",
                    "player_A_id", "player_B_id", "tourney_id",
                    "match_num", "original_winner_id"]

FEATURES_FOR_STANDARD_SCALING = ["rank_point_diff", "rank_diff", "age_diff", "height_diff"]

FEATURES_FOR_ROBUST_SCALING = ["player_A_rank_points", "player_B_rank_points",
                      "player_A_rank", "player_B_rank", "player_A_ht", "player_B_ht", "player_A_seed",
                      "player_B_seed", "draw_size", "player_A_age", "player_B_age"]

FEATURES_TO_RECOMBINE = ["same_hand", "player_A_is_seeded", "player_B_is_seeded"]

# casting int to float to avoid FutureWarning due to setting an item of incompatible dtype
# when scaling.
COLS_TO_CAST_FLOAT = list(set(FEATURES_FOR_STANDARD_SCALING + FEATURES_FOR_ROBUST_SCALING))




class Tennis_preprocessing:
    def __init__(self):
        self.singles_data = Tennis().get_singles()

        # # for possible future reference
        # self.players_data = Tennis().get_players()
        # self.rankings_data = Tennis().get_rankings()

    def duplicates_and_match_stats(self, df) -> pd.DataFrame:
        """
        Method that drops duplicate entries when called on a dataframe that
        contains singles' data. It also drops match statistics that are not
        taken into account for the MVP model.
        """

        # duplicates
        df = df.drop_duplicates()

        # match statistics
        stats_columns = STATS_COLUMNS_TO_DROP
        df = df.drop(stats_columns, axis = 1)

        return df


    def missing_values(self, df) -> pd.DataFrame:
        """
        Method that deals with missing values.
        - Imputes height, age, and draw_size using the median.
        - Imputes entry with 'R'.
        - Imputes rank for unranked players and rank points with 0.
        - Handles non-numeric seeds and imputes missing seeds with rank.
        """

        ############################################################
        # imputing player_ht, player_age and draw_size with median #
        ############################################################

        # instantiating imputer
        height_age_draw_imputer = SimpleImputer(strategy = "median", keep_empty_features = True)

        # fitting and transforming
        df["winner_ht"], df["loser_ht"] = height_age_draw_imputer.fit_transform(df[["winner_ht", "loser_ht"]]).T
        df["winner_age"], df["loser_age"] = height_age_draw_imputer.fit_transform(df[["winner_age", "loser_age"]]).T
        df["draw_size"] = height_age_draw_imputer.fit_transform(df[["draw_size"]]).ravel()

        ##################################
        # imputing player_entry with "R" #
        ##################################

        # instantiating imputer
        entry_imputer = SimpleImputer(strategy = "constant", fill_value = "R", keep_empty_features = True)

        # fitting and transforming
        df["winner_entry"], df["loser_entry"] = entry_imputer.fit_transform(df[["winner_entry", "loser_entry"]]).T

        ###########################################################################################
        # filling player_rank with a rank that is greater than the greatest rank in the dataframe #
        ###########################################################################################

        if not (df["winner_rank"].isna().sum() + df["loser_rank"].isna().sum()):
            max_rank = max(df["winner_rank"].max(), df["loser_rank"].max())

            # instantiating imputer
            rank_imputer = SimpleImputer(strategy = "constant", fill_value = max_rank + 1, keep_empty_features = True)

            # fitting and transforming
            df["winner_rank"], df["loser_rank"] = rank_imputer.fit_transform(df[["winner_rank", "loser_rank"]]).T
        else:
            df["winner_rank"] = df["winner_rank"].fillna(value = 1)
            df["loser_rank"] = df["loser_rank"].fillna(value = 1)


        #################################################################
        # imputing player_ioc, player_hand, score and surface with mode #
        #################################################################

        # instantiating imputer
        hand_surface_score_ioc_imputer = SimpleImputer(strategy = "most_frequent", keep_empty_features = True)

        # fitting and transforming
        df["winner_ioc"], df["loser_ioc"] = hand_surface_score_ioc_imputer.fit_transform(df[["winner_ioc", "loser_ioc"]]).T
        df["winner_hand"], df["loser_hand"] = hand_surface_score_ioc_imputer.fit_transform(df[["winner_hand", "loser_hand"]]).T
        df["surface"], df["score"] = hand_surface_score_ioc_imputer.fit_transform(df[["surface", "score"]]).T



        ##############################################
        # filling unranked player rank points with 0 #
        ##############################################

        # instantiating imputer
        rank_point_imputer = SimpleImputer(strategy = "constant", fill_value = 0, keep_empty_features = True)

        # fitting and transforming
        df["winner_rank_points"], df["loser_rank_points"] = \
                        rank_point_imputer.fit_transform(df[["winner_rank_points", "loser_rank_points"]]).T

        #########################################################
        # converting non-numeric seeds ('Q', 'WC', etc.) to NaN #
        #########################################################
        df['winner_seed'] = pd.to_numeric(df['winner_seed'], errors='coerce')
        df['loser_seed'] = pd.to_numeric(df['loser_seed'], errors='coerce')

        ##############################################
        # creating binary column if player is seeded #
        ##############################################

        df["winner_is_seeded"] = df["winner_seed"].notna().astype(int)
        df["loser_is_seeded"] = df["loser_seed"].notna().astype(int)

        ########################################
        # filling player_seed with player_rank #
        ########################################

        df['winner_seed'] = df['winner_seed'].fillna(df['winner_rank'])
        df['loser_seed'] = df['loser_seed'].fillna(df['loser_rank'])

        return df


    def target_column(self, df, seed = SEED) -> pd.DataFrame:
        """
        Method that turns winner and loser into player_A and player_B.
        Then, randomly interchanges their values and creates a target column
        that contains 0 or 1, depending on who won (player A or player B).

        Finally, drops all the features that will not be used.
        """

        # renaming

        df["original_winner_id"] = df["winner_id"].copy()

        renamed_columns = RENAMED_COLUMNS

        df = df.rename(columns=renamed_columns)

        # swapping
        columns_to_swap = COLUMNS_TO_RANDOM_SWAP

        np.random.seed(seed)
        swap_mask = np.random.choice([True, False], size=len(df))

        for col_A, col_B in zip(columns_to_swap[::2], columns_to_swap[1::2]):
            df.loc[swap_mask, [col_A, col_B]] = df.loc[swap_mask, [col_B, col_A]].values

        # creating target column (0 if player A won, 1 if player B won)
        df["target"] = 0  # initialize all to 0 (player A won)
        df.loc[df["player_A_id"] != df["original_winner_id"], "target"] = 1  # change to 1 if player B won

        # dropping columns that are not to be used in the MVP
        drop_cols = MVP_COLUMNS_TO_DROP

        df = df.drop(columns=drop_cols, errors='ignore')

        return df

    def same_hand(self, df) -> pd.DataFrame:
        """
        Method that creates a binary column that shows whether the 2 players
        use the same hand or not.
        """
        df["same_hand"] = (df["player_A_hand"] == df["player_B_hand"]).astype(int)

        return df

    def create_diffs(self,df) -> pd.DataFrame:
        """
        This method creates differences of features between players, like age, height,
        rank and rank points.
        """

        df["height_diff"] = df["player_A_ht"] - df["player_B_ht"]
        df["age_diff"] = df["player_A_age"] - df["player_B_age"]
        df["rank_diff"] = df["player_A_rank"] - df["player_B_rank"]
        df["rank_point_diff"] = df["player_A_rank_points"] - df["player_B_rank_points"]

        return df

    def clean_singles_MVP_one_df(self, df) -> pd.DataFrame:
        """
        This method applies all cleaning methods to a single df from the singles_data dictionary.
        All these are based on the MVP methodology and subsequently may need tweaking.
        Casts numeric columns intended for scaling to float type to avoid FutureWarning.
        """
        df = self.duplicates_and_match_stats(df)
        df = self.missing_values(df)
        df = self.target_column(df)
        df = self.same_hand(df)
        df = self.create_diffs(df)

        # casting to float
        cols_to_cast = [col for col in COLS_TO_CAST_FLOAT if col in df.columns]

        for col in cols_to_cast:
            if col in df.columns and df[col].dtype != 'object':
                df[col] = df[col].astype(float)

        return df

    def clean_singles_MVP_all_dfs(self) -> dict:
        """
        This method applies all the cleaning and feature engineering methods created so far to each
        of the keys in the self.data dictionary. Returns the cleaned dictionary

        Keep in mind that all these methods are based on the MVP methodology
        and subsequently may need tweaking.
        """

        cleaned_dict = {}

        for key in self.singles_data.keys():
            cleaned_dict[key] = self.clean_singles_MVP_one_df(self.singles_data[key])

        self.singles_data = cleaned_dict

        return self.singles_data

    def create_X_and_y(self,df) -> tuple:
        """
        Separates the df into features and target, returning a tuple whose first
        element is the features as a dataframe and the second is the target as a Series.
        """

        cols = list(df.columns)
        cols.remove('target')
        X = df[cols]
        y = df['target']

        return X, y

    def split_and_scale(self, X, y, test_size = 0.3, random_state = 2010) -> tuple:
        """
        Performs Train/Test split at a ratio of 0.7:0.3 unless otherwise specified.
        Starts with a specific seed unless otherwise specified.

        Then, applies RobustScaler and StandardScaler to the train set only, but transforms both sets
        and returns their scaled versions in the form of a tuple, whose last elementis the scaled features list,
        and the first four are X_train, X_test, y_train and y_test.
        """
        # split
        X_train, X_test, y_train, y_test = train_test_split(X, y, \
                test_size = test_size, random_state = random_state)

        # features to be StandardScaled (they have only a few outliers and are gaussianly distributed)
        feats_to_std_scale = FEATURES_FOR_STANDARD_SCALING

        # features to be RobustScaled (they have outliers and/or non gaussian distribution)
        feats_to_rob_scale = FEATURES_FOR_ROBUST_SCALING

        # features to be Normalized (they are ordinal and don't have outliers)
        feats_to_minmax_scale = []

        # all feats to be scaled, for later use
        scaled_feats = feats_to_std_scale + feats_to_rob_scale + feats_to_minmax_scale

        # ensure columns exist before fitting
        fit_rob_cols = [col for col in feats_to_rob_scale if col in X_train.columns]
        fit_std_cols = [col for col in feats_to_std_scale if col in X_train.columns]

        # copies to avoid warning
        X_train = X_train.copy()
        X_test = X_test.copy()

        # instantiating scalers
        rob_scaler = RobustScaler()
        std_scaler = StandardScaler()

        # fitting to train set ONLY and transforming all of them
        if fit_rob_cols:
            rob_scaler.fit(X_train[fit_rob_cols])
            X_train.loc[:, fit_rob_cols] = rob_scaler.transform(X_train[fit_rob_cols])
            X_test.loc[:, fit_rob_cols] = rob_scaler.transform(X_test[fit_rob_cols])

        if fit_std_cols:
            std_scaler.fit(X_train[fit_std_cols])
            X_train.loc[:, fit_std_cols] = std_scaler.transform(X_train[fit_std_cols])
            X_test.loc[:, fit_std_cols] = std_scaler.transform(X_test[fit_std_cols])

        # resetting index of targets for later use
        y_train = y_train.reset_index(drop = True)
        y_test = y_test.reset_index(drop = True)

        return X_train, X_test, y_train, y_test, scaled_feats


    def encode_and_recombine(self, X_train, X_test, scaled_feats) -> tuple:
        """
        Performs encoding in appropriate features. OneHotEncoder is fitted on train
        but applied on both train and test to transform.

        Creates a new X_train and X_test with encoded features. These are then recombined
        with the scaled numerical features from a previous method and a tuple with the
        recombined version is returned.
        """

        # features to be encoded
        feats_to_encode = X_train.select_dtypes(include = "object").columns.to_list()
        if 'best_of' in X_train.columns and 'best_of' not in feats_to_encode:
             feats_to_encode.append("best_of")

        # instantiating encoder
        # we drop the first category for each feature encoded, to reduce complexity
        # the handle_unknown is useful because there is a chance that there are categories
        # in the test set that are not present in the train set, so it might cause an error
        ohe = OneHotEncoder(sparse_output = False, drop = "first",  handle_unknown='ignore')

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

        # recombining
        X_train_combined = pd.concat([X_train[scaled_feats], X_train_encoded_df, X_train[FEATURES_TO_RECOMBINE]], axis=1)
        X_test_combined = pd.concat([X_test[scaled_feats], X_test_encoded_df, X_test[FEATURES_TO_RECOMBINE]], axis=1)

        return X_train_combined, X_test_combined

    def split_scale_encode_MVP_one_df(self, df) -> tuple:
        """
        Combines methods on splitting, scaling and encoding into one method and is applied to a single df,
        returning a tuple containing the clean and preprocessed X_train, X_test, y_train, y_test
        """
        # could later add an if loop to catch the error if the df has not be cleaned yet.
        # for now we just comment out the cleaning step, we suppose it has already happened
        # df = self.clean_singles_MVP_one_df(df)
        X, y = self.create_X_and_y(df)
        X_train, X_test, y_train, y_test, scaled_feats = self.split_and_scale(X, y)
        X_train_combined, X_test_combined = self.encode_and_recombine(X_train, X_test, scaled_feats)

        return X_train_combined, X_test_combined, y_train, y_test

    def full_preprocess_one_df_MVP(self, df):
        """
        Performs the full cleaning and preprocessing steps to a single df.
        Returns X_train, X_test, y_train, y_test
        """
        cleaned_df = self.clean_singles_MVP_one_df(df)
        X_train, X_test, y_train, y_test = self.split_scale_encode_MVP_one_df(cleaned_df)

        return X_train, X_test, y_train, y_test

    def split_scale_encode_MVP_all_dfs(self):
        """
        Combines all methods on splitting, scaling and encoding and applies them
        to all dfs in the singles_data dictionary.
        *** ASSUMES self.singles_data CONTAINS CLEANED DATAFRAMES ***
        Returns the singles_data dictionary that has as value the list with the preprocessed
        X_train, X_test, y_train, y_test.
        """
        processed_data = {}
        for key, value in self.singles_data.items():
            # assumes 'value' is already clean
            X_train, X_test, y_train, y_test = self.split_scale_encode_MVP_one_df(value)
            processed_data[key] = [X_train, X_test, y_train, y_test]

        self.singles_data = processed_data # Update the instance state
        return self.singles_data


    def full_preprocess_all_dfs_MVP(self):
        """
        Performs the full cleaning and preprocessing steps to all dfs in the singles_data dictionary.
        Returns the singles_data dictionary that has as value the list with the preprocessed
        X_train, X_test, y_train, y_test.
        """
        self.clean_singles_MVP_all_dfs()
        self.split_scale_encode_MVP_all_dfs()
        return self.singles_data
