# Returner and Explorer method using reinforcement learning Generating a Moving Model

![simulator](https://user-images.githubusercontent.com/58085267/142912124-a956c261-7140-44d1-98ff-ec65b0d1090d.gif)

■ taxi 　　★ taxi customer

## What is this?
A simulator that uses reinforcement learning to create a movement model of Returner, a human movement pattern, from cab movement data.
I used the San Francisco cab data I purchased to create a travel model.


## Details
Human mobility is classified into Returner (travelers who visit only the vicinity of frequently visited places) and Explorer (travelers who visit places other than frequently visited places). 
In this research, taxi and road network used in the simulation are created, and the probability distribution of the simulation area is generated from the San Francisco taxi travel data.
The simulation is performed with the created data, and in doing so, the cab takes the optimal action and acquires experience using reinforcement learning. At this time, the ε-greedy method is used to select the next action of the cab from the utilization and search.
As a result of the simulation, we have successfully generated a mobility model based on the data.
