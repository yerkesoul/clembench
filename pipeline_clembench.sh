#!/bin/bash
# Usage: ./pipeline_clembench.sh [default,with_gpt4] [game_name:optional] [--parallel]
# Preparation: ./setup.sh
echo
echo "==================================================="
echo "PIPELINE: Starting $0"
echo "==================================================="
echo
games=("imagegame" "privateshared" "taboo" "referencegame" "wordle" "wordle_withclue" "wordle_withcritic")
models_default_multiplayer=(
  "text-davinci-003--text-davinci-003"
  "gpt-3.5-turbo--gpt-3.5-turbo"
  "claude-v1.3--claude-v1.3"
  "luminous-supreme--luminous-supreme"
)
# shellcheck disable=SC2034
models_default_singleplayer=(
  "text-davinci-003"
  "gpt-3.5-turbo"
  "claude-v1.3"
  "luminous-supreme"
)
models_gpt4_multiplayer=("gpt-4--gpt-4" "gpt-4--gpt-3.5-turbo" "gpt-3.5-turbo--gpt-4")
# shellcheck disable=SC2034
models_gpt4_singleplayer=("gpt-4")
# Check if the user provided an argument
if [ $# -eq 0 ]; then
  echo "Please provide one of [default,with_gpt4] as the first argument."
  exit 1
fi

# Flag to track if --parallel is given (note that this might cause rate limit errors)
do_parallel=false
for arg in "$@"; do
  if [ "$arg" == "--parallel" ]; then
    do_parallel=true
    break
  fi
done

# Check if the user provided an optional argument
if [ $# -eq 2 ]; then
  games=("$2")
fi
echo "Run games: (${games[*]})"
# Determine which list to use based on the argument
if [ "$1" == "default" ]; then
  if [ ! -f "key_default.json" ]; then
    echo "File 'key_default.json' does not exist. Aborting."
    exit 1
  fi
  cp key_default.json key.json
elif [ "$1" == "with_gpt4" ]; then
  if [ ! -f "key_gpt4.json" ]; then
    echo "File 'key_gpt4.json' does not exist. Aborting."
    exit 1
  fi
  cp key_gpt4.json key.json
else
  echo "Invalid argument. Please choose 'default' or 'with_gpt4'."
  exit 1
fi
# Run the benchmark (each game in parallel using screen sessions)
for game in "${games[@]}"; do
  if [ "$1" == "default" ]; then
    if [ "$game" == "privateshared" ] || [ "$game" == "wordle" ] || [ "$game" == "wordle_withclue" ]; then
      models=("${models_default_singleplayer[@]}")
    else
      models=("${models_default_multiplayer[@]}")
    fi
  elif [ "$1" == "with_gpt4" ]; then
    if [ "$game" == "privateshared" ] || [ "$game" == "wordle" ] || [ "$game" == "wordle_withclue" ]; then
      models=("${models_gpt4_singleplayer[@]}")
    else
      models=("${models_gpt4_multiplayer[@]}")
    fi
  else
    echo "Invalid argument. Please choose 'default' or 'with_gpt4'."
    exit 1
  fi
  model_pairs=("${models[@]}")
  for model_pair in "${model_pairs[@]}"; do
    if $do_parallel; then
      screen_name="${game}.${model_pair}"
      if screen -ls | grep -q "$screen_name" >/dev/null; then
        screen -S "$screen_name" -X kill
        echo "Restart screen $screen_name"
      else
        echo "Starting screen ${screen_name}"
      fi
      screen -dmS "${screen_name}" bash -c "./run.sh ${game} ${model_pair}"
    else
      ./run.sh "${game}" "${model_pair}"
    fi
  done
done
echo "========================================================================="
echo "All sessions started. Wait for the runtime.<game>.<model_pair>.log files."
echo "You can use 'killall screen' before running this script again."
echo "========================================================================="
