# clembench: A Framework for the Systematic Evaluation of Chat-Optimized Language Models as Conversational Agents

The cLLM (chat-optimized Large Language Model, "clem") framework tests such models' ability to engage in games – rule-constituted activities played using language.
The framework is a systematic way of probing for the situated language understanding of language using agents.

This repository contains the code for setting up the framework and implements a number of games that are further discussed in 

> Chalamalasetti, K., Götze, J., Hakimov, S., Madureira, B., Sadler, P., & Schlangen, D. (2023). clembench: Using Game Play to Evaluate Chat-Optimized Language Models as Conversational Agents (arXiv:2305.13455). arXiv. https://doi.org/10.48550/arXiv.2305.13455

### Results Overview

For each model (pairing), shows how many games were played to completion (%
played), an indicator of rule-following capabilities. “qlty score” indicates how well the completed games were
played (higher is better, max is 100). all is the average over all games scores, the remaining columns show results
broken down by game (averaged over all episodes)

<table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>model</th>
      <th>game:</th>
      <th>all</th>
      <th>drawing</th>
      <th>priv/sh</th>
      <th>reference</th>
      <th>taboo</th>
      <th>wordle</th>
      <th>wordle+cl</th>
      <th>wordle+cl+cr</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">lm/lm</th>
      <th>% played</th>
      <td>16.67</td>
      <td>0.0 </td>
      <td>0.0 </td>
      <td>0.0 </td>
      <td>0.0 </td>
      <td>100.0 </td>
      <td>6.67 </td>
      <td>10.0 </td>
    </tr>
    <tr>
      <th>qlty score</th>
      <td>0.00</td>
      <td>/</td>
      <td>/</td>
      <td>/</td>
      <td>/</td>
      <td>0.0 </td>
      <td>0.0 </td>
      <td>0.0 </td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">cl/cl</th>
      <th>% played</th>
      <td>63.81</td>
      <td>0.0 </td>
      <td>100.0 </td>
      <td>100.0 </td>
      <td>0.0 </td>
      <td>100.0 </td>
      <td>96.67 </td>
      <td>50.0 </td>
    </tr>
    <tr>
      <th>qlty score</th>
      <td>43.77</td>
      <td>/</td>
      <td>60.27 (24.65)</td>
      <td>88.89 (31.87)</td>
      <td>/</td>
      <td>0.0 </td>
      <td>40.8 (46.42)</td>
      <td>28.89 (36.44)</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">3/3</th>
      <th>% played</th>
      <td>47.86</td>
      <td>65.0 </td>
      <td>10.0 </td>
      <td>83.33 </td>
      <td>23.33 </td>
      <td>73.33 </td>
      <td>46.67 </td>
      <td>33.33 </td>
    </tr>
    <tr>
      <th>qlty score</th>
      <td>38.09</td>
      <td>44.77 (25.59)</td>
      <td>55.32 (19.65)</td>
      <td>63.33 (49.01)</td>
      <td>14.29 (37.8)</td>
      <td>0.0 </td>
      <td>46.43 (41.44)</td>
      <td>42.5 (50.07)</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">3.5/3.5</th>
      <th>% played</th>
      <td>87.98</td>
      <td>97.5 </td>
      <td>85.0 </td>
      <td>100.0 </td>
      <td>56.67 </td>
      <td>100.0 </td>
      <td>90.0 </td>
      <td>86.67 </td>
    </tr>
    <tr>
      <th>qlty score</th>
      <td>36.02</td>
      <td>65.41 (25.32)</td>
      <td>56.72 (27.75)</td>
      <td>66.67 (47.81)</td>
      <td>29.41 (46.97)</td>
      <td>0.0 </td>
      <td>18.52 (39.58)</td>
      <td>15.38 (26.21)</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">3.5/4</th>
      <th>% played</th>
      <td><strong>94.03</strong></td>
      <td>97.5</td>
      <td>/</td>
      <td>100.0 </td>
      <td>86.67 </td>
      <td>100.0 </td>
      <td>90.0 </td>
      <td>90.0 </td>
    </tr>
    <tr>
      <th>qlty score</th>
      <td>34.68</td>
      <td>70.28 (24.08)</td>
      <td>/</td>
      <td>72.22 (45.43)</td>
      <td>28.85 (40.43)</td>
      <td>0.0 </td>
      <td>18.52 (39.58)</td>
      <td>18.21 (29.51)</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">4/3.5</th>
      <th>% played</th>
      <td>87.78</td>
      <td>80.0 </td>
      <td>/</td>
      <td>100.0 </td>
      <td>46.67 </td>
      <td>100.0 </td>
      <td>100.0 </td>
      <td>100.0 </td>
    </tr>
    <tr>
      <th>qlty score</th>
      <td>56.01</td>
      <td>79.47 (23.93)</td>
      <td>/</td>
      <td>61.11 (49.44)</td>
      <td>96.43 (13.36)</td>
      <td>3.56 (9.55)</td>
      <td>47.06 (42.27)</td>
      <td>48.44 (45.27)</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">4/4</th>
      <th>% played</th>
      <td>93.81</td>
      <td>80.0 </td>
      <td>100.0 </td>
      <td>100.0 </td>
      <td>76.67 </td>
      <td>100.0 </td>
      <td>100.0 </td>
      <td>100.0 </td>
    </tr>
    <tr>
      <th>qlty score</th>
      <td><strong>60.59</strong></td>
      <td>92.09 (12.04)</td>
      <td>78.55 (9.74)</td>
      <td>77.78 (42.16)</td>
      <td>73.19 (37.18)</td>
      <td>4.56 (10.59)</td>
      <td>47.89 (41.55)</td>
      <td>50.11 (44.98)</td>
    </tr>
  </tbody>
</table>

Date of data collection: 19.05.2023 (v1)

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
