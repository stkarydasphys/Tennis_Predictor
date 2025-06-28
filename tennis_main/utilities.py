"""
This module contains various functions that are used for simple tasks throughout
"""

def score_parser(strng):
    """
    Parses the score to generate tiebreaks.
    """
    if not isinstance(strng, str):
        return 0, 0, 0

    scores = strng.split()
    tiebreaks = [sc for sc in scores if "(" in sc]
    tie_breaks_happened = len(tiebreaks)
    tie_breaks_winner = 0
    tie_breaks_loser = 0

    for tb in tiebreaks:
        if tb[0] > tb[2]:
            tie_breaks_winner +=1
        else:
            tie_breaks_loser += 1

    return tie_breaks_winner, tie_breaks_loser, tie_breaks_happened
