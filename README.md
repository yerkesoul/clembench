# clembench: A Framework for the Systematic Evaluation of Chat-Optimized Language Models as Conversational Agents

The cLLM (chat-optimized Large Language Model, "clem") framework tests such models' ability to engage in games – rule-constituted activities played using language.
The framework is a systematic way of probing for the situated language understanding of language using agents.

This repository contains the code for setting up the framework and implements a number of games that are further discussed in 

> Chalamalasetti, K., Götze, J., Hakimov, S., Madureira, B., Sadler, P., & Schlangen, D. (2023). clembench: Using Game Play to Evaluate Chat-Optimized Language Models as Conversational Agents (arXiv:2305.13455). arXiv. https://doi.org/10.48550/arXiv.2305.13455

## Evaluation Results

### Overall Results

We have evaluated the following models and games:

<table border="1" class="dataframe">
    <thead>
    <tr style="text-align: right;">
        <th></th>
        <th>model</th>
        <th>claude-v1.3-t0.0--claude-v1.3-t0.0</th>
        <th>gpt-3.5-turbo-t0.0--gpt-3.5-turbo-t0.0</th>
        <th>gpt-3.5-turbo-t0.0--gpt-4-t0.0</th>
        <th>gpt-4-t0.0--gpt-3.5-turbo-t0.0</th>
        <th>gpt-4-t0.0--gpt-4-t0.0</th>
        <th>luminous-supreme-t0.0--luminous-supreme-t0.0</th>
        <th>text-davinci-003-t0.0--text-davinci-003-t0.0</th>
    </tr>
    <tr>
        <th>activity</th>
        <th>metric</th>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
    </tr>
    </thead>
    <tbody>
    <tr>
        <th rowspan="3" valign="top">imagegame</th>
        <th>Aborted Ratio</th>
        <td>nan (nan)</td>
        <td>nan (nan)</td>
        <td>0.3 (0.46)</td>
        <td>0.94 (0.23)</td>
        <td>0.94 (0.23)</td>
        <td>nan (nan)</td>
        <td>nan (nan)</td>
    </tr>
    <tr>
        <th>F1</th>
        <td>0.0 (0.0)</td>
        <td>0.47 (0.37)</td>
        <td>0.52 (0.4)</td>
        <td>0.03 (0.15)</td>
        <td>0.06 (0.23)</td>
        <td>0.0 (0.0)</td>
        <td>0.34 (0.29)</td>
    </tr>
    <tr>
        <th>Success Ratio</th>
        <td>nan (nan)</td>
        <td>nan (nan)</td>
        <td>0.1 (0.3)</td>
        <td>0.0 (0.0)</td>
        <td>0.06 (0.23)</td>
        <td>nan (nan)</td>
        <td>nan (nan)</td>
    </tr>
    <tr>
        <th rowspan="4" valign="top">privateshared</th>
        <th>Aborted Ratio</th>
        <td>nan (nan)</td>
        <td>nan (nan)</td>
        <td>nan (nan)</td>
        <td>nan (nan)</td>
        <td>0.0 (0.0)</td>
        <td>nan (nan)</td>
        <td>nan (nan)</td>
    </tr>
    <tr>
        <th>Kappa</th>
        <td>0.49 (0.26)</td>
        <td>0.42 (0.21)</td>
        <td>nan (nan)</td>
        <td>nan (nan)</td>
        <td>0.68 (0.12)</td>
        <td>0.2 (0.16)</td>
        <td>0.12 (0.21)</td>
    </tr>
    <tr>
        <th>Slot-Filling-Accuracy</th>
        <td>0.94 (0.13)</td>
        <td>0.98 (0.06)</td>
        <td>nan (nan)</td>
        <td>nan (nan)</td>
        <td>0.95 (0.11)</td>
        <td>0.77 (0.31)</td>
        <td>0.96 (0.08)</td>
    </tr>
    <tr>
        <th>Success Ratio</th>
        <td>nan (nan)</td>
        <td>nan (nan)</td>
        <td>nan (nan)</td>
        <td>nan (nan)</td>
        <td>0.0 (0.0)</td>
        <td>nan (nan)</td>
        <td>nan (nan)</td>
    </tr>
    <tr>
        <th rowspan="2" valign="top">referencegame</th>
        <th>Aborted Ratio</th>
        <td>nan (nan)</td>
        <td>nan (nan)</td>
        <td>0.0 (0.0)</td>
        <td>0.0 (0.0)</td>
        <td>0.0 (0.0)</td>
        <td>nan (nan)</td>
        <td>nan (nan)</td>
    </tr>
    <tr>
        <th>Success Ratio</th>
        <td>nan (nan)</td>
        <td>nan (nan)</td>
        <td>0.67 (0.48)</td>
        <td>0.61 (0.49)</td>
        <td>0.75 (0.44)</td>
        <td>nan (nan)</td>
        <td>nan (nan)</td>
    </tr>
    <tr>
        <th rowspan="2" valign="top">taboo</th>
        <th>Speed</th>
        <td>4.0 (0.0)</td>
        <td>3.5 (1.14)</td>
        <td>3.17 (1.23)</td>
        <td>2.6 (1.52)</td>
        <td>2.6 (1.43)</td>
        <td>4.0 (0.0)</td>
        <td>3.9 (0.55)</td>
    </tr>
    <tr>
        <th>Success Ratio</th>
        <td>nan (nan)</td>
        <td>nan (nan)</td>
        <td>0.33 (0.48)</td>
        <td>0.47 (0.51)</td>
        <td>0.53 (0.51)</td>
        <td>nan (nan)</td>
        <td>nan (nan)</td>
    </tr>
    <tr>
        <th rowspan="2" valign="top">wordle</th>
        <th>Aborted Ratio</th>
        <td>0.03 (0.18)</td>
        <td>0.0 (0.0)</td>
        <td>0.0 (0.0)</td>
        <td>0.0 (0.0)</td>
        <td>0.0 (0.0)</td>
        <td>0.0 (0.0)</td>
        <td>0.43 (0.5)</td>
    </tr>
    <tr>
        <th>Success Ratio</th>
        <td>0.0 (0.0)</td>
        <td>0.0 (0.0)</td>
        <td>0.0 (0.0)</td>
        <td>0.17 (0.38)</td>
        <td>0.2 (0.41)</td>
        <td>0.0 (0.0)</td>
        <td>0.0 (0.0)</td>
    </tr>
    <tr>
        <th rowspan="2" valign="top">wordle_withclue</th>
        <th>Aborted Ratio</th>
        <td>0.03 (0.18)</td>
        <td>0.17 (0.38)</td>
        <td>0.1 (0.31)</td>
        <td>0.03 (0.18)</td>
        <td>0.0 (0.0)</td>
        <td>0.93 (0.25)</td>
        <td>0.63 (0.49)</td>
    </tr>
    <tr>
        <th>Success Ratio</th>
        <td>0.5 (0.51)</td>
        <td>0.27 (0.45)</td>
        <td>0.13 (0.35)</td>
        <td>0.67 (0.48)</td>
        <td>0.67 (0.48)</td>
        <td>0.0 (0.0)</td>
        <td>0.23 (0.43)</td>
    </tr>
    <tr>
        <th rowspan="2" valign="top">wordle_withcritic</th>
        <th>Aborted Ratio</th>
        <td>0.53 (0.51)</td>
        <td>0.23 (0.43)</td>
        <td>0.2 (0.41)</td>
        <td>0.0 (0.0)</td>
        <td>0.0 (0.0)</td>
        <td>0.9 (0.31)</td>
        <td>0.73 (0.45)</td>
    </tr>
    <tr>
        <th>Success Ratio</th>
        <td>0.23 (0.43)</td>
        <td>0.17 (0.38)</td>
        <td>0.3 (0.47)</td>
        <td>0.6 (0.5)</td>
        <td>0.63 (0.49)</td>
        <td>0.0 (0.0)</td>
        <td>0.2 (0.41)</td>
    </tr>
    </tbody>
</table>

### Game details

- A Simple Word Game: [taboo](docs/taboo.md)
- A Word-Guessing Game Based on Clues: [wordle](docs/wordle.md)
- Drawing Instruction Giving and Following: [image](docs/image.md)
- An ASCII Picture Reference Game: [reference](docs/reference.md)
- Scorekeeping: [private and shared](docs/privateshared.md)

## Using the benchmark

We welcome you to contribute to or extend the benchmark with your own games and models. 
Please simply open a pull request. You can find more information on how to use the benchmark in the links below.

- [How to run the benchmark and evaluation locally](docs/howto_run_benchmark.md)
- [How to add a new model as a backend](docs/howto_add_backend.md)
- [How to add and run your own game](docs/howto_add_games.md)
- [How to integrate with Slurk](docs/howto_slurk.md)
