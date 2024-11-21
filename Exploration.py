
"""

Yeah, aspects of that were okay but I think it was too complex and also too simple at the same time. 

Maybe the threat appraisal process is messy - you work with limited information and make predictions based on what you sense and what you've connected those senses to before. 

So there might be some basic activation tendencies, like that f function from earlier, I liked that. 

But then working with the threat perception and appraisal - maybe certain dimensions of a perceived threat (qualities whatever) are used to predict other qualities, which are then more tightly linked to the situation requiring or not Fight/flight/freeze, based on past experience/memory. 

So that's to say that there's a learning process - threat vectors can be relearned and reappraised - e.g. spider doesn't always have to trigger whatever it is that leads to fight or flight (idk, maybe people around you screaming and encouraging you to kill a spider). Like ultimately there are some threat triggers that probably just trigger action, but there are some that don't and have varying degrees of proximity to a threshold trigger. 

So maybe repeated exposure is a means of relearning the association between threat aspects and triggering threats. 

Also I think deliberately resolving the threat without fight/flight/freeze (e.g. calmly picking up a spider and setting it down) should count as a non-threatening resolution
- This goes against the amygdala threat appraisal indicating that high levels of activation were required, but I suspect this doesn't strongly indicate that no activation was necessary 
- It's like you take a chance on doing something that is not FFF, and if that goes well, you get some amygdala relearning/deactivation
- If you don't take a chance on it, you still have the chance to learn that you were appraising the threat incorrectly - the threat wasn't the trigger itself but you predicted a trigger based on threats
- you can learn to predict triggers better 



There’s 
Activation A, 
Baseline B, 
Appraisal App, 
Threat vector T, 
T_obs, 
an innate appraisal f, 
an adaptive appraisal g, g should max out eventually
and g’s parameters theta, limit to weight given to a set of features
Also a threshold for exploration thresh



Working on G 
- threats can be explored, which can result in a variety of outcomes
- fight has some exploratory qualities to it 
- successful exploration should lower the activation for G, increase threshold
- failed exploration should increase the activation for G, decrease threshold

- flight and freeze are avoidant and probably leave appraisal high, maybe reinforce appraisal where it is, maybe modify threshold to reinforce avoidant behavior
- if they fail I think it's technically an exploration, which can results in varied outcomes





Issues:

Need G

"""

# Reinitializing the simulation after reset

import numpy as np
import matplotlib.pyplot as plt

# Parameters for simulation
n_exposures = 50  # Number of threat encounters
k = 5  # Sensitivity of exploration probability to threshold
threshold_explore = 0.5  # Initial exploration threshold
alpha_avoid = 0.02  # Decrease in threshold due to avoidance
beta_explore = 0.05  # Increase in threshold due to exploration
learning_rate = 0.1  # Learning rate for reinforcement (not directly used here)
alpha_explore_fail = 0.05  # Decrease in threshold due to failed exploration
P_success_explore = 0.8  # Probability that exploration succeeds
P_success_avoid = 0.95  # Probability that avoidance succeeds


# Generate random threat appraisals (fixed for consistency)
# np.random.seed(42)  # For reproducibility
A_appraisal = np.random.uniform(0.3, 1.0, n_exposures)  # Appraised threat levels

# Initialize tracking variables
exploration_probs = []
thresholds = [threshold_explore]
exploration_outcomes = []

for exposure in range(n_exposures):
    # Calculate exploration probability
    P_explore = 1 / (1 + np.exp(k * (A_appraisal[exposure] - threshold_explore)))
    exploration_probs.append(P_explore)
    
    # Decide whether to explore or avoid
    explored = np.random.rand() < P_explore
    exploration_outcomes.append(explored)
    
    # Update thresholds based on outcome
    if explored:
        # Determine if exploration succeeds
        success = np.random.rand() < P_success_explore
        if success:
        	threshold_explore += beta_explore  # Successful exploration increases threshold
        else:
        	threshold_explore -= alpha_explore_fail # failed exploration
    else:
        # Determine if avoidance succeeds
        success = np.random.rand() < P_success_avoid
        if success:
        	threshold_explore -= alpha_avoid  # Avoidance decreases threshold
        else: # Accidentally explore
        	success = np.random.rand() < P_success_explore 
	        if success:
	        	threshold_explore += beta_explore  # Successful exploration increases threshold
	        if not success:
	        	threshold_explore -= alpha_explore_fail #failed exploration
    
    # Track threshold
    thresholds.append(threshold_explore)

# Plot results
fig, axs = plt.subplots(2, 1, figsize=(10, 8))

# Plot exploration probabilities over exposures
axs[0].plot(range(n_exposures), exploration_probs, label="Exploration Probability")
axs[0].set_title("Exploration Probability Over Exposures")
axs[0].set_xlabel("Exposure Number")
axs[0].set_ylabel("Probability")
axs[0].legend()

# Plot threshold dynamics over exposures
axs[1].plot(range(n_exposures + 1), thresholds, label="Threshold (Explore)", linestyle="--", color="orange")
axs[1].set_title("Exploration Threshold Over Exposures")
axs[1].set_xlabel("Exposure Number")
axs[1].set_ylabel("Threshold")
axs[1].legend()

plt.tight_layout()
plt.show()


