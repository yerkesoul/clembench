# The Lightweight Dialogue Game framework

## How to add and run your own game

Games are interactions between two agents that are administrated by a game master.
The game master handles all game logic and all messages are passed via the game master.
The game master evaluates whether an agent has made a valid move, when the dialog ends, and what evaluation measures are logged.

### The GameMaster

The game master is implemented as a `GameMaster` class.
The `GameMaster` does

- access all relevant resources via `load_file()` or `load_json()`
- `setup()` the concrete games instances
- coordinate the `play()` of the game instances (up to multiple episodes)
- record the game episodes
- call a `GameEvaluator` to `evaluate()` the game records XXX still-correct?

The `GameMaster` must implement the following methods:

1. `_on_setup(self, **kwargs)`: A method that instantiates a particular game episode that is described by the game instance (the `kwargs`)

For example, the taboo game is set up as a game between two players, with a maximum number of turns and a list of target words and related words. The game state also includes whether a description was valid and a whether a guess was correct or not:
  ```python
    def _on_setup(self, **game_instance):
        self.game_instance = game_instance

        self.describer = WordDescriber(self.player_backends[0], self.max_turns)
        self.guesser = WordGuesser(self.player_backends[1])

        self.add_player(self.describer)
        self.add_player(self.guesser)

        self.target_word = game_instance["target_word"]
        self.related_words = game_instance["related_word"]

        self.invalid_response = False
        self.target_in_description = False
        self.guess_word = None

  ```

2. `_does_game_proceed(self)`: A method that returns `True` if the game should proceed at a particular game state or `False` if not.

For example, the taboo game ends when one of the players gave an invalid response (e.g. a response that does not adhere to the expected format), when the describer used the target word in the description, when the guess was correct or when the maximum number of turns was reached:
```python
    def _does_game_proceed(self):
        if self.invalid_response:
            return False
        if self.target_in_description:
            return False  # stop game if clue is wrong (for now)
        if self.guess_word == self.target_word:
            return False
        if self.current_turn >= self.max_turns:
            return False
        return True

```

### The Player

A `Player` object receives `messages` and returns a textual response.
A player generates this response either as a `_api_response()`
(calling a deployed cLLM) or by implemented behavior in `_static_response()`.

For example, the taboo game guesser agent can be implemented as a player that can be a cLLM with a static response that always guesses the word "pear":

```python
from clemgame.clemgame import Player

class WordGuesser(Player):

   def __init__(self, model_name):
      super().__init__(model_name)

   def _static_response(self, messages, turn_idx):
      # mock response
      return f'Pear'
```

### Generating game instances

In order to let agents play a game, you need a description that instantiate single episodes.
For example, in the taboo game, each episode is played with a specific target word that also comes with a list of other, related and forbidden words.

The clemgame framework provides a `GameInstanceGenerator` class that you can use to generate full instances that also include initial prompts for the models and other meta information for running experiments.

For example, in the taboo game, we
- use word lists of 3 different frequency levels low/medium/high
- want to test 3 LLMs (taboo is played between 2 cLLMs)
- we fix the maximum number of turns to `N_GUESSES`
- we generate a fixed number of instances, `N_INSTANCES`
```python
from clemgame.clemgame import GameInstanceGenerator

N_INSTANCES = 20  # how many different target words; zero means "all"
N_GUESSES = 3  # how many tries the guesser will have
N_REATED_WORDS = 3
LANGUAGE = "en"

class TabooGameInstanceGenerator(GameInstanceGenerator):

    def __init__(self):
        super().__init__("taboo")

    def on_generate(self):
        player_assignments = list(itertools.permutations([OpenAI.MODEL_GPT_35, Anthropic.MODEL_CLAUDE_13]))
        for difficulty in ["low", "medium", "high"]:

            # first choose target words based on the difficultly
            fp = f"resources/target_words/{LANGUAGE}/{difficulty}_freq_100"
            target_words = self.load_file(file_name=fp, file_ending=".txt").split('\n')
            if N_INSTANCES > 0:
                assert len(target_words) >= N_INSTANCES, \
                    f'Fewer words available ({len(target_words)}) than requested ({N_INSTANCES}).'
                target_words = random.sample(target_words, k=N_INSTANCES)

            # use the same target_words for the different player assignments
            experiment = self.add_experiment(f"{difficulty}_{LANGUAGE}", dialogue_partners=player_assignments)
            experiment["max_turns"] = N_GUESSES

            describer_prompt = self.load_template("resources/initial_prompts/initial_describer")
            guesser_prompt = self.load_template("resources/initial_prompts/initial_guesser")
            experiment["describer_initial_prompt"] = describer_prompt
            experiment["guesser_initial_prompt"] = guesser_prompt

            for game_id in tqdm(range(len(target_words))):
                target = target_words[game_id]

                game_instance = self.add_game_instance(experiment, game_id)
                game_instance["target_word"] = target
                game_instance["related_word"] = []

                if len(game_instance["related_word"]) < N_REATED_WORDS:
                    print(f"Found less than {N_REATED_WORDS} related words for: {target}")
```

This will then generate game instances as a json file at `games/taboo/in/instances.json`

### Adding your own game

To add your own game, create a submodule in `games` with the name of your game, for example `games.hellogame`.

Add to the module a `master.py` that implements the `GameMaster`.

### Running experiments with your game

```
python3 scripts/cli.py -m gpt-3.5-turbo [-e greet_en] hellogame
```

Note: With -e you can specify specific experiments to run.

This will create a records folder in your game directory as the following:

```
records
└── gpt-3.5-turbo
    └── greet_en
        ├── episode_0
        │ ├── instance_0.json
        │ ├── interaction.json
        │ └── transcript.html
        ├── episode_1
        │ ├── instance_1.json
        │ ├── interaction.json
        │ └── transcript.html
        │ ...
        └── experiment_greet_en.json
```

The top level is `records` followed by directories that mention the involved model (pairings).

The model (pairing) sub-folders will contain a directory structure for each experiment
and the experiments episodes (game plays).

The episodes are defined by the game instances (from the `instances.json`) and
contain the instance parameters `instance_n.json`, an `interaction.json` and a nice human-viewable `transcript.html`.

The experiment folder also contains a `experiment_name.json` that contains the run parameters.

# Troubleshooting

### AssertionError: messages history must not be empty for Player

When using the `DialogueGameMaster`, then here the framework prevents a call to the remote API with an empty message
history.

1. Maybe you forgot to add the initial prompt to the players messages in `_on_before_game()`.
   For this use `self.add_user_message(<player>, prompt)`

2. You forgot to add the response of the preceding player to the
   message history of the current player in `_after_add_player_response(other_player, utt)`.
   For this use `self.add_user_message(current_player, utt)`
