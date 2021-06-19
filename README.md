# TargetExposerBot

This is an implementation of a SLAM target searching algorithm.

# The algorithm

The algorithm is based on bayas probabillity formulas.
It has 2 Options of running:
  * static agent
  * dynamiv agent
  
The static agent scans for targets while remaining in starting position.
The dynamic agent scans for targets while moving in space, towords probability COG, one step in each iteration.
 
 
Params for running the algorithm:
```
algo_args = dict(agent_location=(0, 0), # Agent's starting coordinate
                 alpha=0.5, # Rate of false alarms
                 init_prob_value=0.5, probability starting value
                 p_ta=1.0, # True alarm probability
                 probability_to_match=0.95, # Indicator for a success in finding the targets
                 size=20, # Size of the board, assuming board is symetrical
                 sensor_info=10, # Factor unit for the sensor and env
                 targets_locations=[(0, 9), (4, 14), (15, 8)]) # Targets positions
```



