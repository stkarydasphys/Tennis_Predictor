"""
Script that contains tests for the methods that extract player stats
"""
# to run, while being in the Tennis Predictor folder, type:
# python tests/player_stats_test.py

# preliminaries
import pandas as pd
import numpy as np

# project related
from tennis_main.player_stats_new import Player
from tennis_main.utilities import score_parser

def make_dummy_data():
    """
    Creates a dummy DataFrame to simulate player stats to be used in tests.
    """
    data = {
        "winner_id": [1, 2, 1, 3],
        "loser_id": [2, 1, 3, 1],
        "tourney_name": ["Tournament A", "Tournament B", "Tournament A", "Tournament C"],
        "tourney_level": ["Grand Slam", "Masters", "Grand Slam", "Challenger"],
        "tourney_surface": ["Hard", "Clay", "Hard", "Grass"],
        "tourney_id": ["T1", "T2", "T1", "T3"],
        "tourney_date": [20240101, 20240102, 20240101, 20240103],
        "match_num": [1, 1, 2, 1],
        "score": ["6-4 7-6(5)", "3-6 6-3 7-6(3)", "6-2 6-4", "7-5 6-3"],
        "w_svpt": [50, 60, 55, 65],
        "l_svpt": [45, 55, 50, 60],
        "w_1stIn": [30, 40, 35, 45],
        "l_1stIn": [28, 38, 32, 42],
        "w_df": [2, 3, 1, 4],
        "l_df": [1, 2, 0, 3],
        "w_ace": [5, 6, 4, 7],
        "l_ace": [3, 4, 2, 5],
        "w_1stWon": [20, 25, 22, 30],
        "l_1stWon": [18, 22, 16, 20],
        "w_2ndWon": [10, 12, 11, 15],
        "l_2ndWon": [9, 11, 8, 10],
        "w_bpFaced": [4, 5, 3, 6],
        "l_bpFaced": [3, 4, 2, 5],
        "w_bpSaved": [2, 3, 1, 4],
        "l_bpSaved": [1, 2, 0, 3],
        "w_SvGms": [10, 12, 11, 13],
        "l_SvGms": [9, 11, 10, 12],
    }

    return pd.DataFrame(data)

def test_create_2nd_serve_in():
    """
    Tests the create_2nd_serve_in method of Player class
    1. Checks if the columns 'w_2ndIn' and 'l_2ndIn' are created
    2. Validates the calculation of 'w_2ndIn' and 'l_2ndIn'
    3. Ensures the method handles edge cases
    4. Verifies the method works with a dummy DataFrame
    """
    # creating a dummy player and DataFrames
    player = Player()
    df = make_dummy_data()

    # creating a DataFrame for edge case with zero stats
    zero_stats_edge_case_data = {
        "winner_id": [1],
        "loser_id": [2],
        "tourney_name": ["Tournament A"],
        "tourney_level": ["Grand Slam"],
        "tourney_surface": ["Hard"],
        "tourney_id": ["T1"],
        "tourney_date": [20240101],
        "match_num": [1],
        "score": ["6-4 7-6(5)"],
        "w_svpt": [0],
        "l_svpt": [0],
        "w_1stIn": [0],
        "l_1stIn": [0],
        "w_df": [0],
        "l_df": [0],
        "w_ace": [0],
        "l_ace": [0],
        "w_1stWon": [0],
        "l_1stWon": [0],
        "w_2ndWon": [0],
        "l_2ndWon": [0],
        "w_bpFaced": [0],
        "l_bpFaced": [0],
        "w_bpSaved": [0],
        "l_bpSaved": [0],
        "w_SvGms": [0],
        "l_SvGms": [0],
    }

    edge_case_df = pd.DataFrame(zero_stats_edge_case_data)

    # calling the method to test
    result = player.create_2nd_serve_in(df)
    edge_case_result = player.create_2nd_serve_in(edge_case_df)

    ################
    # Basic checks #
    ################

    # checking if a dataframe is returned and if the columns are created
    assert isinstance(result, pd.DataFrame), "Result should be a DataFrame"
    assert not result.empty, "Result DataFrame should not be empty"

    assert "w_2ndIn" in result.columns, "w_2ndIn column is not in result dataframe"
    assert "l_2ndIn" in result.columns, "l_2ndIn column is not in result dataframe"


    # checking if the method returns the expected columns
    expected_columns = [
        "winner_id", "loser_id", "tourney_name", "tourney_level", "tourney_surface",
        "tourney_id", "tourney_date", "match_num", "score", "w_svpt", "l_svpt",
        "w_1stIn", "l_1stIn", "w_df", "l_df", "w_ace", "l_ace",
        "w_1stWon", "l_1stWon", "w_2ndWon", "l_2ndWon",
        "w_bpFaced", "l_bpFaced", "w_bpSaved", "l_bpSaved",
        "w_SvGms", "l_SvGms", "w_2ndIn", "l_2ndIn"
    ]
    assert all(col in result.columns for col in expected_columns), "Result DataFrame should contain all expected columns"
    assert len(result.columns) == len(expected_columns), "Result DataFrame should have the correct number of columns"

    # checking if the result has the same number of rows as the input DataFrame
    assert len(result) == len(df), "Result DataFrame should have the same number of rows as input DataFrame"

    # checking calculation for all rows of the dummy DataFrame
    expected_w_2ndIn = df["w_svpt"] - df["w_1stIn"] - df["w_df"]
    expected_l_2ndIn = df["l_svpt"] - df["l_1stIn"] - df["l_df"]
    assert np.all(result["w_2ndIn"] >= 0), "w_2ndIn should be non-negative"
    assert np.all(result["l_2ndIn"] >= 0), "l_2ndIn should be non-negative"
    assert np.all(result["w_2ndIn"] == expected_w_2ndIn), "w_2ndIn calculation is incorrect"
    assert np.all(result["l_2ndIn"] == expected_l_2ndIn), "l_2ndIn calculation is incorrect"

    ##############
    # Edge cases #
    ##############

    # checking if edge case dataframe is returned and if the columns are created
    assert isinstance(edge_case_result, pd.DataFrame), "Edge case result should be a DataFrame"
    assert not edge_case_result.empty, "Edge case result DataFrame should not be empty"

    assert "w_2ndIn" in edge_case_result.columns, "w_2ndIn column is not in edge case result dataframe"
    assert "l_2ndIn" in edge_case_result.columns, "l_2ndIn column is not in edge case result dataframe"

    # checking if the edge case DataFrame has the expected columns
    assert all(col in edge_case_result.columns for col in expected_columns), "Edge case result DataFrame should contain all expected columns"
    assert len(edge_case_result.columns) == len(expected_columns), "Edge case result DataFrame should have the correct number of columns"

    # checking if the edge case result has the correct number of rows
    assert len(edge_case_result) == len(edge_case_df), "Edge case result DataFrame should have the same number of rows as input DataFrame"

    # checking if the edge case result has 0 for 'w_2ndIn' and 'l_2ndIn'
    assert np.all(edge_case_result["w_2ndIn"] == 0), "w_2ndIn should be 0 for edge case"
    assert np.all(edge_case_result["l_2ndIn"] == 0), "l_2ndIn should be 0 for edge case"


def test_create_tiebreaks():
    """
    Tests the create_tiebreaks method of Player class
    1. Checks if the columns w_tiebreaks_won, w_tiebreaks_happened, w_tiebreaks_lost,
       l_tiebreaks_won, l_tiebreaks_lost, l_tiebreaks_happened are created
    2. Validates the calculation of these columns
    3. Ensures the method handles edge cases
    4. Verifies the method works with a dummy DataFrame
    """
    # creating a dummy player and DataFrames
    player = Player()
    df = make_dummy_data()

    # creating a DataFrame for edge case with zero stats
    data_without_tiebreaks = {
        "winner_id": [1],
        "loser_id": [2],
        "tourney_name": ["Tournament A"],
        "tourney_level": ["Grand Slam"],
        "tourney_surface": ["Hard"],
        "tourney_id": [101],
        "tourney_date": [pd.Timestamp("2023-01-01")],
        "match_num": [1],
        "score": ["6-4 6-4"],
        "w_svpt": [50],
        "l_svpt": [40],
        "w_1stIn": [30],
        "l_1stIn": [20],
        "w_df": [5],
        "l_df": [3],
        "w_ace": [10],
        "l_ace": [8],
        "w_1stWon": [25],
        "l_1stWon": [15],
        "w_2ndWon": [15],
        "l_2ndWon": [10],
        "w_bpFaced": [5],
        "l_bpFaced": [3],
        "w_bpSaved": [4],
        "l_bpSaved": [2],
        "w_SvGms": [6],
        "l_SvGms": [4],
        "w_tiebreaks_won": [0],
        "l_tiebreaks_won": [0],
        "w_tiebreaks_happened": [0],
        "l_tiebreaks_happened": [0],
        "w_tiebreaks_lost": [0],
        "l_tiebreaks_lost": [0]
    }

    edge_case_df = pd.DataFrame(data_without_tiebreaks)

    # calling the method to test
    result = player.create_tiebreaks(df)
    edge_case_result = player.create_tiebreaks(edge_case_df)

    ################
    # Basic checks #
    ################

    # checking if a dataframe is returned and if the columns are created
    assert isinstance(result, pd.DataFrame), "Result should be a DataFrame"
    assert not result.empty, "Result DataFrame should not be empty"

    assert "w_tiebreaks_won" in result.columns, "w_tiebreaks_won column is not in result dataframe"
    assert "l_tiebreaks_won" in result.columns, "l_tiebreaks_won column is not in result dataframe"
    assert "w_tiebreaks_happened" in result.columns, "w_tiebreaks_happened column is not in result dataframe"
    assert "l_tiebreaks_happened" in result.columns, "l_tiebreaks_happened column is not in result dataframe"
    assert "w_tiebreaks_lost" in result.columns, "w_tiebreaks_lost column is not in result dataframe"
    assert "l_tiebreaks_lost" in result.columns, "l_tiebreaks_lost column is not in result dataframe"

    # checking if the method returns the expected columns
    expected_columns = [
        "winner_id", "loser_id", "tourney_name", "tourney_level", "tourney_surface",
        "tourney_id", "tourney_date", "match_num", "score", "w_svpt", "l_svpt",
        "w_1stIn", "l_1stIn", "w_df", "l_df", "w_ace", "l_ace",
        "w_1stWon", "l_1stWon", "w_2ndWon", "l_2ndWon",
        "w_bpFaced", "l_bpFaced", "w_bpSaved", "l_bpSaved",
        "w_SvGms", "l_SvGms", "w_tiebreaks_won", "l_tiebreaks_won",
        "w_tiebreaks_happened", "l_tiebreaks_happened", "w_tiebreaks_lost", "l_tiebreaks_lost"
    ]
    assert all(col in result.columns for col in expected_columns), "Result DataFrame should contain all expected columns"
    assert len(result.columns) == len(expected_columns), "Result DataFrame should have the correct number of columns"

    # checking if the result has the same number of rows as the input DataFrame
    assert len(result) == len(df), "Result DataFrame should have the same number of rows as input DataFrame"

    # checking calculation for all rows of the dummy DataFrame
    expected_w_tiebreaks_won, expected_l_tiebreaks_won, expected_tiebreaks_happened = zip(*df["score"].apply(score_parser))
    # note: score_parser returns a tuple of (w_tiebreaks_won, l_tiebreaks_won, tiebreaks_happened)
    # which originally is a tuple of integers. When applied to a Series, it returns a Series of tuples,
    # so we need to unpack them into separate variables before performing vectorized calculations
    expected_w_tiebreaks_won = np.array(expected_w_tiebreaks_won)
    expected_l_tiebreaks_won = np.array(expected_l_tiebreaks_won)
    expected_tiebreaks_happened = np.array(expected_tiebreaks_happened)

    expected_w_tiebreaks_lost = expected_tiebreaks_happened - expected_w_tiebreaks_won
    expected_l_tiebreaks_lost = expected_tiebreaks_happened - expected_l_tiebreaks_won
    assert np.all(result["w_tiebreaks_won"] >= 0), "w_tiebreaks_won should be non-negative"
    assert np.all(result["l_tiebreaks_won"] >= 0), "l_tiebreaks_won should be non-negative"
    assert np.all(result["w_tiebreaks_happened"] >= 0), "w_tiebreaks_happened should be non-negative"
    assert np.all(result["l_tiebreaks_happened"] >= 0), "l_tiebreaks_happened should be non-negative"
    assert np.all(result["w_tiebreaks_lost"] >= 0), "w_tiebreaks_lost should be non-negative"
    assert np.all(result["l_tiebreaks_lost"] >= 0), "l_tiebreaks_lost should be non-negative"
    assert np.all(result["w_tiebreaks_won"] == expected_w_tiebreaks_won), "w_tiebreaks_won calculation is incorrect"
    assert np.all(result["l_tiebreaks_won"] == expected_l_tiebreaks_won), "l_tiebreaks_won calculation is incorrect"
    assert np.all(result["w_tiebreaks_happened"] == expected_tiebreaks_happened), "w_tiebreaks_happened calculation is incorrect"
    assert np.all(result["l_tiebreaks_happened"] == expected_tiebreaks_happened), "l_tiebreaks_happened calculation is incorrect"
    assert np.all(result["w_tiebreaks_lost"] == expected_w_tiebreaks_lost), "w_tiebreaks_lost calculation is incorrect"
    assert np.all(result["l_tiebreaks_lost"] == expected_l_tiebreaks_lost), "l_tiebreaks_lost calculation is incorrect"

    ##############
    # Edge cases #
    ##############

    # checking if edge case dataframe is returned and if the columns are created
    assert isinstance(edge_case_result, pd.DataFrame), "Edge case result should be a DataFrame"
    assert not edge_case_result.empty, "Edge case result DataFrame should not be empty"

    assert "w_tiebreaks_won" in edge_case_result.columns, "w_tiebreaks_won column is not in edge result dataframe"
    assert "l_tiebreaks_won" in edge_case_result.columns, "l_tiebreaks_won column is not in edge result dataframe"
    assert "w_tiebreaks_happened" in edge_case_result.columns, "w_tiebreaks_happened column is not in edge result dataframe"
    assert "l_tiebreaks_happened" in edge_case_result.columns, "l_tiebreaks_happened column is not in edge result dataframe"
    assert "w_tiebreaks_lost" in edge_case_result.columns, "w_tiebreaks_lost column is not in edge result dataframe"
    assert "l_tiebreaks_lost" in edge_case_result.columns, "l_tiebreaks_lost column is not in edge result dataframe"

    # checking if the edge case DataFrame has the expected columns
    assert all(col in edge_case_result.columns for col in expected_columns), "Edge case result DataFrame should contain all expected columns"
    assert len(edge_case_result.columns) == len(expected_columns), "Edge case result DataFrame should have the correct number of columns"

    # checking if the edge case result has the correct number of rows
    assert len(edge_case_result) == len(edge_case_df), "Edge case result DataFrame should have the same number of rows as input DataFrame"

    # checking if the edge case result has 0 for 'w_tiebreaks_won' and 'l_tiebreaks_won'
    assert np.all(edge_case_result["w_tiebreaks_won"] == 0), "w_tiebreaks_won should be 0 for edge case"
    assert np.all(edge_case_result["l_tiebreaks_won"] == 0), "l_tiebreaks_won should be 0 for edge case"
    assert np.all(edge_case_result["w_tiebreaks_happened"] == 0), "w_tiebreaks_happened should be 0 for edge case"
    assert np.all(edge_case_result["l_tiebreaks_happened"] == 0), "l_tiebreaks_happened should be 0 for edge case"
    assert np.all(edge_case_result["w_tiebreaks_lost"] == 0), "w_tiebreaks_lost should be 0 for edge case"
    assert np.all(edge_case_result["l_tiebreaks_lost"] == 0), "l_tiebreaks_lost should be 0 for edge case"






# running the test

if __name__ == "__main__":
    test_create_2nd_serve_in()
    test_create_tiebreaks()
    print("All tests passed.")
