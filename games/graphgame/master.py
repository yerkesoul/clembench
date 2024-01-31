from typing import Dict, Tuple, List
import json
import numpy as np
import ast
from clemgame.clemgame import GameMaster, GameBenchmark, Player, DialogueGameMaster
from clemgame.metrics import METRIC_ABORTED, METRIC_SUCCESS, METRIC_LOSE, METRIC_REQUEST_COUNT, \
    METRIC_REQUEST_COUNT_VIOLATED, METRIC_REQUEST_COUNT_PARSED, METRIC_REQUEST_SUCCESS
from clemgame import get_logger
from clemgame import file_utils, string_utils

import random
GAME_NAME = "graphgame"

INVALID = np.nan

logger = get_logger(__name__)


class PathGuesser(Player):

    def __init__(self, model_name):
        super().__init__(model_name)
        
    def _custom_response(self, messages, turn_idx):
        # mock response
        random_path = random.choice(["North", "South", "East", "West"])
        return f' Instruction: go to {random_path}'

class PathDescriber(Player):

    def __init__(self, model_name,max_turns):
        super().__init__(model_name)
        self.max_turns = max_turns

    def _custom_response(self, messages, turn_idx):
        return f"This is the message {messages}"
            
    def get_directions(node, direction_list):
        node_directions = None  
        for i in direction_list:
            if i[0]==node:
                node_directions=i[1]
                break
        return node_directions
    
    def string_available_directions(word_list):
        return ', '.join(word_list)

    def clear_utterance(utterance):

        utterance = utterance.replace("Instruction:", "")
        utterance = utterance.replace("go", "")
        utterance = string_utils.remove_punctuation(utterance)
        return utterance
    
    def have_common_element(str1, str2):
        common_elements = ["east", "west", "north", "south"]
        
        # Convert strings to sets of words
        words1 = set(str1.split())
        words2 = set(str2.split())
        
        # Check for exact matches
        common_matches = words1.intersection(words2).intersection(common_elements)
        
        # Return True if there is at least one exact match
        return any(match in common_elements for match in common_matches)
    

    def check_path_answer(utterance: str, directions: List[str], node) -> List[Dict]:

        utterance = PathDescriber.clear_utterance(utterance)
        previous_direction= PathDescriber.get_directions(node, directions)
        previous_dirrection_changed=PathDescriber.string_available_directions(previous_direction) 
        previous_dirrection_no_pq=string_utils.remove_punctuation(previous_dirrection_changed)

        if not PathDescriber.have_common_element(utterance, previous_dirrection_no_pq):
            return [{
            "message": f"The desired direction is not in available paths:'{previous_dirrection_changed}'",
            "type": 0}]
        
    def get_nextnode_label(moves, node, utterance):

        utterance = PathDescriber.clear_utterance(utterance)
        for move in moves:
            if move["node"]==node:
                moves_node=move['node_moves']
                for step in moves_node:
                    if step[0]==utterance:
                        next_label=step[1]
        return next_label


class GraphGame(DialogueGameMaster):
    """
    This class implements a graph traversal game in which player A (DecisionMaker). 
    """

    def __init__(self, experiment: Dict, player_backends: List[str]):
        super().__init__(GAME_NAME, experiment, player_backends)
        self.steps_made=0
        self.max_turns=30
        self.visited_nodes=[]

    def _on_setup(self, **game_instance):
        logger.info("_on_setup")
        self.game_instance = game_instance
        self.invalid_response = False
        self.game_error = None
        self.game_stop=False
        self.directions_next_node= None
        self.chosen_direction=None
        
        self.nodes= ast.literal_eval(game_instance['Graph_Nodes'])
        self.edges= ast.literal_eval(game_instance['Graph_Edges'])
        self.playerA_initial_prompt= game_instance["player_1_prompt_header"]
        self.initial_position= ast.literal_eval(game_instance["Current_Position"])
        self.directions= ast.literal_eval(game_instance['Directions'])
        self.moves=ast.literal_eval(game_instance['Moves'])
        self.initial_directions= PathDescriber.get_directions(self.initial_position, self.directions)
        self.changed_initial_directions= PathDescriber.string_available_directions(self.initial_directions)
        self.playerA_initial_prompt = self.playerA_initial_prompt.replace("$INITIAL_DIRECTIONS$",self.changed_initial_directions)
        self.directions_next_node= self.changed_initial_directions
        self.visited_nodes.append(self.initial_position)
        self.guesser = PathGuesser(self.player_backends[0])
        self.describer = PathDescriber(self.player_backends[1], self.max_turns)

        self.add_player(self.guesser)
        self.add_player(self.describer)

    def _on_before_game(self):
        
        self.add_user_message(self.guesser, self.playerA_initial_prompt)

    def _does_game_proceed(self):
        "Proceed untill all nodes arent visited"
        if self.invalid_response:
            self.log_to_self("invalid format", "abort game")
            return False
        if self.game_stop:
            self.log_to_self("Game stopped","The guesser decided to stop the game")
            False
        if self.game_error is not None:
            error_type = self.game_error["type"]
            if error_type == 0:
                self.log_to_self("Direction not available", "The desired direction is not in available paths")
            return False  #stop game if clue is wrong (for now)
        if self.current_turn >= self.max_turns:
            self.log_to_self("max turns reached", str(self.max_turns))
            return False
        return True


    def _validate_player_response(self, player: Player, utterance: str) -> bool:

        if player == self.guesser:
            if not utterance.startswith("Instruction:"):
                self.invalid_response = True
                return False
            if "done" in utterance.lower():
                self.game_stop=True

        if player == self.describer:
            if not utterance.startswith("Directions:"):
                self.invalid_response = True
                return False
            "Check if the direction is valid"
            the_last_node=self.visited_nodes[-1]
            errors = PathDescriber.check_path_answer(utterance, self.directions, the_last_node)
            if errors:
                error = errors[0]
                self.game_error = error
                return False
            else:
                next_node_label=PathDescriber.get_nextnode_label(self.moves, the_last_node,utterance)
                if next_node_label in self.nodes:
                    self.visited_nodes.append(next_node_label)
                    list_directions_nextnode= PathDescriber.get_directions( next_node_label, self.directions)
                    self.directions_next_node=PathDescriber.string_available_directions(list_directions_nextnode)

        self.log_to_self("valid format", "continue")
        return True

    def _on_parse_response(self, player: Player, utterance: str) -> Tuple[str, bool]:
        "Decide if a response utterance should be modified. If not simply return the utterance."
        "When a modified utterance and a true value is returned, then a 'parse' event is logged."
        
        if player == self.describer:
            utterance = utterance.replace("Directions:", "")
            utterance = utterance.strip()
            self.log_to_self("directions", utterance)
        if player == self.guesser:
            utterance = utterance.replace("Instruction:", "")
            utterance = utterance.strip()
            utterance = utterance.lower()
            utterance = string_utils.remove_punctuation(utterance)
            self.chosen_direction = utterance.lower()
            self.log_to_self("instruction", self.chosen_direction)
        return utterance, False

    def _after_add_player_response(self, player: Player, utterance: str):
        """Add the utterance to other player's history, if necessary.
        To do this use the method add_user_message(other_player,utterance)."""

        if player == self.describer:
            utterance = f"Directions:{self.directions_next_node}."
            self.add_user_message(self.guesser, utterance)

        if player == self.guesser:
            utterance=PathDescriber.clear_utterance(utterance)
            utterance = f"Instruction: go {utterance}."
            self.add_user_message(self.describer, utterance)


    def compute_scores(self, episode_interactions: Dict) -> None:
        """ Episode level scores"""
        turn_scores = []
        invalid_response = False  # Note: This only takes into consideration that both players were compliant or not
        all_nodes_visited=False
        for turn_idx, turn in enumerate(episode_interactions["turns"]):
            for event in turn:
                action = event["action"]
                if action["type"] == "invalid format":
                    invalid_response = True
                if action["type"] == "Game stopped":
                    if set(self.visited_nodes)==set(self.nodes):
                        all_nodes_visited = True
        if invalid_response: 
            self.log_episode_score(METRIC_ABORTED, 1)
            self.log_episode_score(METRIC_SUCCESS, 0)
        else:
            self.log_episode_score(METRIC_ABORTED, 0)
            if all_nodes_visited:
                self.log_episode_score(METRIC_SUCCESS, 1)

                                   
class GraphGameBenchmark(GameBenchmark):

    def __init__(self):
        super().__init__(GAME_NAME)

    def get_description(self):
        return "Graph Game."

    def create_game_master(self, experiment: Dict, player_backends: List[str]) -> GameMaster:
        return GraphGame(experiment, player_backends)


def main():
    # select one experiment and instance
    experiments = file_utils.load_json("in/instances.json", "graphgame")
    experiment_1 = experiments["experiments"][0]
    game_1 = experiment_1["game_instances"]
    master = GraphGame(experiment_1, ["mock", "mock"])
    master.setup(**game_1)
    master.play()


if __name__ == '__main__':
    main()
