# Running the benchmark

## Setting up

### Install dependencies

```
pip install -r requirements.txt
```

### API Key

Create a file `key.json` in the project root and paste in your api key (and organisation optionally).

```
{
  "openai": {
            "organisation": "<value>", 
            "api_key": "<value>"
            },
  "anthropic": {
            "api_key": "<value>"
            },
  "alephalpha": {
            "api_key": "<value>"
            }
}
```

Note: You can look up your api key for OpenAI at https://platform.openai.com/account/api-keys and for Anthoropic
at https://console.anthropic.com/account/keys, AlephAlpha can be found
here: https://docs.aleph-alpha.com/docs/introduction/luminous/

### Available models

Currently available values are:

- `"gpt-4"`
- `"gpt-3.5-turbo"`
- `"text-davinci-003"`
- `"claude-v1.3"`
- `"claude-v1.3-100k"`
- `"luminous-supreme-control"`
- `"luminous-supreme"`
- `"luminous-extended"`
- `"luminous-base"`
- `"google/flan-t5-xxl"`

Models can be added in `clemgame/api.py`.


## Validating your installation

Add keys to the API providers as described above.

Go into the project root and prepare path to run from cmdline

```
source prepare_path.sh
```

Then run the cli script

```
python3 scripts/cli.py -m gpt-3.5-turbo--gpt-3.5-turbo run taboo
```

(The `-m` option tells the script which model to use; since taboo is a two player game, we need both partners to be specified here.)

This should give you an output that contains something like the following:
```
Playing games: 100%|██████████████████████████████████| 20/20 [00:48<00:00,  2.41s/it]
```

If that is the case, output (transcripts of the games played) will have been written to `results/taboo` (in the main directory of the code).

Unfortunately, at the moment the code fails silently for example if model names are wrong, so make sure that you see the confirmation that the game actually has been played.

You can get more information about what you can do with the `cli` script via:

```
python3 scripts/cli.py --help
```

For example, you can use that script to get a more readable version of the game play jsons like so:

```
python3 scripts/cli.py transcribe taboo
```

(The `results` directory will now hold html and LaTeX views of the transcripts.)


## Running the benchmark

Go into the project root and prepare path to run from cmdline

```
source prepare_path.sh
```

Then, run the wrapper script:

```
./pipeline_clembench.sh
```

Internally, this uses `run.sh` to run individual game/model combinations. Inspect the code to see how things are done.



## Running the evaluation

All details from running the benchmarked are logged in the respective game directories,
with the format described in ```logdoc.md```.

We provide an evaluation script at `evaluation/basiceval.py` that produces a number of tables and visualizations for the benchmark. New models (their name abbreviation), metrics (their range) and game/model (their order) must be added manually to the constants in ```evaluation/evalutils.py```.
