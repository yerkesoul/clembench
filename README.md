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
      <th></th>
      <th>all</th>
      <th>taboo</th>
      <th>wordle</th>
      <th>wordle_withclue</th>
      <th>wordle_withcritic</th>
      <th>imagegame</th>
      <th>referencegame</th>
      <th>privateshared</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">lm/lm (0.00) </th>
      <th>% played</th>
      <td>16.24</td>
      <td>0.0 </td>
      <td>100.0 </td>
      <td>3.33 </td>
      <td>10.34 </td>
      <td>0.0 </td>
      <td>0.0 </td>
      <td>0.0 </td>
    </tr>
    <tr>
      <th>qlty score</th>
      <td>0.00</td>
      <td>/</td>
      <td>0.0 (0.0)</td>
      <td>0.0 (-)</td>
      <td>0.0 (0.0)</td>
      <td>/</td>
      <td>/</td>
      <td>/</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">ko/ko (1.47) </th>
      <th>% played</th>
      <td>14.76</td>
      <td>0.0 </td>
      <td>86.67 </td>
      <td>16.67 </td>
      <td>0.0 </td>
      <td>0.0 </td>
      <td>0.0 </td>
      <td>0.0 </td>
    </tr>
    <tr>
      <th>qlty score</th>
      <td>10.00</td>
      <td>/</td>
      <td>0.0 (0.0)</td>
      <td>20.0 (44.72)</td>
      <td>/</td>
      <td>/</td>
      <td>/</td>
      <td>/</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">flc/flc (0.71) </th>
      <th>% played</th>
      <td>0.95</td>
      <td>0.0 </td>
      <td>0.0 </td>
      <td>3.33 </td>
      <td>3.33 </td>
      <td>0.0 </td>
      <td>0.0 </td>
      <td>0.0 </td>
    </tr>
    <tr>
      <th>qlty score</th>
      <td>75.00</td>
      <td>/</td>
      <td>/</td>
      <td>50.0 (-)</td>
      <td>100.0 (-)</td>
      <td>/</td>
      <td>/</td>
      <td>/</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">ost/ost (1.73) </th>
      <th>% played</th>
      <td>20.85</td>
      <td>0.0 </td>
      <td>100.0 </td>
      <td>16.67 </td>
      <td>14.29 </td>
      <td>0.0 </td>
      <td>15.0 </td>
      <td>0.0 </td>
    </tr>
    <tr>
      <th>qlty score</th>
      <td>8.33</td>
      <td>/</td>
      <td>0.0 (0.0)</td>
      <td>0.0 (0.0)</td>
      <td>0.0 (0.0)</td>
      <td>/</td>
      <td>33.33 (51.64)</td>
      <td>/</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">vcn/vcn (4.24) </th>
      <th>% played</th>
      <td>13.58</td>
      <td>5.08 </td>
      <td>56.67 </td>
      <td>13.33 </td>
      <td>20.0 </td>
      <td>0.0 </td>
      <td>0.0 </td>
      <td>0.0 </td>
    </tr>
    <tr>
      <th>qlty score</th>
      <td>31.25</td>
      <td>100.0 (0.0)</td>
      <td>0.0 (0.0)</td>
      <td>25.0 (50.0)</td>
      <td>0.0 (0.0)</td>
      <td>/</td>
      <td>/</td>
      <td>/</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">cl/cl (37.06) </th>
      <th>%played</th>
      <td>74.76</td>
      <td>76.92 </td>
      <td>100.0 </td>
      <td>100.0 </td>
      <td>46.43 </td>
      <td>0.0 </td>
      <td>100.0 </td>
      <td>100.0 </td>
    </tr>
    <tr>
      <th>qlty score</th>
      <td>49.58</td>
      <td>68.75 (38.71)</td>
      <td>0.0 (0.0)</td>
      <td>30.56 (40.13)</td>
      <td>30.77 (48.04)</td>
      <td>/</td>
      <td>82.5 (38.48)</td>
      <td>84.87 (18.87)</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">3/3 (15.77) </th>
      <th>% played</th>
      <td>44.50</td>
      <td>28.81 </td>
      <td>66.67 </td>
      <td>36.67 </td>
      <td>23.33 </td>
      <td>57.5 </td>
      <td>82.5 </td>
      <td>16.0 </td>
    </tr>
    <tr>
      <th>qlty score</th>
      <td>35.46</td>
      <td>76.47 (43.72)</td>
      <td>1.25 (5.59)</td>
      <td>31.36 (38.99)</td>
      <td>50.0 (50.0)</td>
      <td>38.7 (27.78)</td>
      <td>36.36 (48.85)</td>
      <td>14.1 (25.21)</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">3.5/3.5 (37.02) </th>
      <th>% played</th>
      <td>85.86</td>
      <td>69.49 </td>
      <td>100.0 </td>
      <td>93.33 </td>
      <td>76.67 </td>
      <td>97.5 </td>
      <td>100.0 </td>
      <td>64.0 </td>
    </tr>
    <tr>
      <th>qlty score</th>
      <td>43.12</td>
      <td>71.95 (44.79)</td>
      <td>0.0 (0.0)</td>
      <td>28.57 (46.0)</td>
      <td>13.19 (30.16)</td>
      <td>60.28 (25.95)</td>
      <td>55.0 (50.38)</td>
      <td>72.83 (13.07)</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">3.5/4 (42.39) </th>
      <th>% played</th>
      <td>86.75</td>
      <td>69.49 </td>
      <td>/</td>
      <td>/</td>
      <td>80.0 </td>
      <td>97.5 </td>
      <td>100.0 </td>
      <td>/</td>
    </tr>
    <tr>
      <th>qlty score</th>
      <td>48.87</td>
      <td>62.6 (45.15)</td>
      <td>/</td>
      <td>/</td>
      <td>10.42 (17.42)</td>
      <td>64.95 (25.45)</td>
      <td>57.5 (50.06)</td>
      <td>/</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">4/3.5 (55.61) </th>
      <th>% played</th>
      <td>82.78</td>
      <td>66.1 </td>
      <td>/</td>
      <td>/</td>
      <td>100.0 </td>
      <td>65.0 </td>
      <td>100.0 </td>
      <td>/</td>
    </tr>
    <tr>
      <th>qlty score</th>
      <td>67.19</td>
      <td>93.59 (23.45)</td>
      <td>/</td>
      <td>/</td>
      <td>46.67 (42.92)</td>
      <td>81.0 (21.54)</td>
      <td>47.5 (50.57)</td>
      <td>/</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">4/4 (59.48) </th>
      <th>% played</th>
      <td>96.06</td>
      <td>94.92 </td>
      <td>100.0 </td>
      <td>100.0 </td>
      <td>100.0 </td>
      <td>77.5 </td>
      <td>100.0 </td>
      <td>100.0 </td>
    </tr>
    <tr>
      <th>qlty score</th>
      <td>61.93</td>
      <td>76.19 (37.45)</td>
      <td>3.67 (8.4)</td>
      <td>49.67 (42.09)</td>
      <td>49.11 (38.46)</td>
      <td>89.06 (22.28)</td>
      <td>75.0 (43.85)</td>
      <td>90.79 (8.2)</td>
    </tr>
  </tbody>
</table>

Date of data collection: June 5 - June 14, 2023 (v2)

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
