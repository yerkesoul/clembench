# common names
METRIC_ABORTED = "Aborted"
"""
At the episode level, either 0 or 1 whether the game play has been aborted (1) or not (0) 
(due to violation of the game rules e.g. not parsable response or re-prompt for n turns)) 
(this metric does not include games lost). Record level: episode
"""

METRIC_LOSE = "Lose"
"""
At the episode level, either 0 or 1 whether the game play has been successful (0) or not (1) 
(this metric does not include aborted games; the game is lost, when the game goal is not reached 
within the declared number of max_turns, in this sense it’s the opposite of success). Record level: episode
"""

METRIC_SUCCESS = "Success"
"""
At the episode level, either 0 or 1 whether the game play has been successful (1) or not (0) 
(this metric does not include aborted games; the game is successful, when the game goal is reached 
within the declared number of max_turns, in this sense it’s the opposite of lost). Record level: episode
"""

METRIC_REQUEST_COUNT = "Request Count"
METRIC_REQUEST_COUNT_PARSED = "Parsed Request Count"
METRIC_REQUEST_COUNT_VIOLATED = "Violated Request Count"
METRIC_REQUEST_SUCCESS = "Request Success Ratio"

BENCH_SCORE = 'Main Score'

METRIC_PLAYED = 'Played'  # 1 - ABORTED
