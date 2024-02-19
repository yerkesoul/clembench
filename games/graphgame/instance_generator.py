import sys
from clemgame.clemgame import GameInstanceGenerator
import ast
import clemgame
logger = clemgame.get_logger(__name__)
game_name="BackwardForwardMoves"

class GraphGameInstanceGenerator(GameInstanceGenerator):

    def __init__(self, player_a_prompt_header, prompt_question, file_graphs):
        super().__init__(game_name)
        self.player_a_prompt_header = player_a_prompt_header
        self.prompt_question = prompt_question
        self.file_graphs = file_graphs

    def on_generate(self):
        player_a_prompt_header = self.load_template(str(self.player_a_prompt_header))
        prompt_question = self.load_template(str(self.prompt_question))
        file= open(str(self.file_graphs), 'r')
        grids=[]
        c=0
        for line in file:
            line=line.rstrip()
            doc=ast.literal_eval(line)
            if c<10:
                grids.append(doc)
            c+=1
        file.close()

        experiment = self.add_experiment(game_name)

        for grid_index,grid in enumerate(grids):

            game_instance = self.add_game_instance(experiment, grid_index)
            game_instance["player_1_prompt_header"] = player_a_prompt_header
            game_instance["player_1_question"] = prompt_question
            game_instance['Graph_Nodes'] = str(grid['Graph_Nodes'])
            game_instance['Graph_Edges'] = str(grid['Graph_Edges'])
            game_instance['Current_Position'] = str(grid['Initial_Position'])
            game_instance['Picture_Name'] = grid['Picture_Name']
            game_instance["Directions"] = str(grid["Directions"])
            game_instance["Moves"] = str(grid["Moves"])
            game_instance["Grid_Dimension"] = str(grid["Grid_Dimension"])
            game_instance['Cycle'] = grid['Cycle']
        print("done")

