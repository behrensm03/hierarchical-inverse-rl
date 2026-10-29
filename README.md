# Hierarchical Inverse Reinforcement Learning for Norm-Adhering Shopping

A course project (AI ethics, Tufts University, Fall 2025) that investigates whether an agent can learn ethical, norm-adhering shopping behavior from expert demonstrations alone, without a hand-written reward function, in a long-horizon task.

**Team:** [Michael Behrens, Kelsey Zeng, Sanjana Sunder, Rahul Jamdade]

## Built on Proper Shopper

This project is an extension of the **Proper Shopper** environment, which is course-provided code written by Teah Markstone, Daniel Kasenberg, Matthias Scheutz, and Vasanth Sarathy. The original environment provides the supermarket simulation, the shopping task, the norms, and the game mechanics. See [`ENVIRONMENT_README.md`](ENVIRONMENT_README.md) for the original documentation and credits.

On top of that environment, we added:

- A **Q-learning expert** (carried over from individual assignments) that generates demonstrations.
- A **hierarchical IRL (HIRL) pipeline** that learns a reward function and policy from those demonstrations.
- An **experiment harness** for evaluating the pipeline against baselines.

## Motivation

Hand-designing reward functions is laborious, tends to be brittle, and bakes in the designer's own interpretation of what proper behavior is. Inverse reinforcement learning (IRL) instead infers a reward function from observed expert behavior. But standard IRL struggles on long-horizon tasks, especially when reward arrives only at the end.

We asked two questions:

1. Can an IRL agent learn ethical shopping behavior from expert demonstrations in this environment?
2. Can it handle random shopping lists autonomously while adhering to norms?

> **Important scope note:** the pipeline is trained per shopping list and requires expert demonstrations of that specific list. It can only perform lists it has seen demonstrated, and it does not generalize to new lists. Our focus was reproducing observed behavior while adhering to norms, not generalization. See [Limitations](#limitations-and-future-work).

## Task

Given a list of 3 items, the agent must pick up a basket, walk to each of the 3 shelves and take the item, then go to the register and pay, all without violating store norms.

## Approach

1. **Generate demonstrations.** The Q-learning expert runs each shopping list 5 times with epsilon = 0.2 action noise, plus extra noise on the recorded x/y coordinates, so demonstrations take varied paths (without it, demos can be identical at certain subgoals).
2. **Segment into subgoals.** Each demonstration is analyzed with a Bayesian mixture model over sliding windows of states. Where locally linear behavior breaks down, we mark a subgoal (these tend to be corners where the expert changes direction). A dynamic-programming step then assigns noisy demonstration states to subgoals and splits the demos accordingly.
3. **Learn each subtask.** A separate maximum-entropy IRL agent is trained on each subgoal.
4. **Chain.** To perform the full task, each subgoal agent generates its portion of the trajectory, and the pieces are stitched together.

We also trained a **single-agent max-ent IRL baseline** on the full task with an enriched feature set (distances to the basket, items, and register, plus flags for holding the basket, each item, and having paid). It heads straight for the final location, never learns the required ordering, and fails. Even though it has extra features compared to our HIRL agent which hint at subgoals, they do not prescribe a required subgoal ordering, which the agent fails to grasp.

## Experiment

- 50 random shopping lists of 3 items each.
- The expert is run per list with epsilon = 0.2 (the demonstrations the learner sees) and once more with epsilon = 0 as a best-case reference.
- The baseline IRL agent generates one trajectory per list.
- The HIRL pipeline is trained separately on each list, then samples 10 trajectories with epsilon = 0.05 and one greedy trajectory (epsilon = 0).
- All trajectories are executed in the supermarket environment and scored on task success (every item taken and paid for) and violation-free rate (no norm violations).

## Results

| Agent | Task success | Violation-free |
|---|---|---|
| Expert (deterministic, upper bound) | 100% | 100% |
| Expert demonstrations (noisy; what the learner sees) | 100% | 71% |
| Single-agent max-ent IRL (baseline) | 0% | 0% |
| HIRL pipeline, epsilon = 0.05 | 59% | 22% |
| **HIRL pipeline, greedy** | **96%** | **92%** |

- Hierarchy is what makes this work: the single-agent baseline fails completely (about 21 violations per run on average, up to 37), while the greedy HIRL agent succeeds on nearly every list.
- The greedy HIRL agent is cleaner (92% violation-free) than the noisy demonstrations it learned from (71%).
- It is also efficient: greedy HIRL stays within about 20 steps on average of the best-case expert.
- Adding even 5% random actions causes a sharp drop. Failures fall into two buckets, not collecting all items or not paying, and both trace to one cause: for speed, the IRL stage *predicts* the next state with a simple function (step size added to current coordinates) instead of executing the action in the environment and observing the resulting state, and that prediction function ignores obstacles, for simplicity. An action that bumps into a shelf leaves the agent believing it is somewhere it is not, because the prediction function returns the updated coordinates as if the move succeeded, but in the environment, the agent actually bounces off the obstacle and remains in the same position, causing a discrepancy that persists for the remainder of the run. Even in failed runs the agent usually completes intermediate goals (on average 1.8 of 3 items in the basket in runs that failed to pay).

## Simplifications to the environment

- Agent step size set to 0.25, so positions are discretized to the nearest 0.25.
- Turning mechanics removed from the learned agent's model; a wrapper turns the agent as needed before it moves or interacts. (In the original environment, an agent that faces east and executes a north move does not move but only turns to face north, and a subsequent north move would actually move the agent. we removed this.)
- Only shelf items are used (no counters).

## Limitations and future work

- **No generalization across lists.** The pipeline is retrained from scratch for every shopping list, and it needs expert demonstrations of that exact list. Without demos, it has no way to produce the subgoals for a new list, so in practice it can only perform tasks an expert has already demonstrated. This is the biggest limitation of the project. A possible way to relax it is described under future work below.
- **Handling lists not seen in training.** Subgoals come from a small vocabulary ("navigate to a location," "pick up an item", etc.), and the shopping task always follows the same pattern: basket, get items, register. A higher-level planner could build the subgoal sequence for a new list, select a previously learned reward parameter for each subgoal type (for example, a navigation policy defined on the displacement between the agent and its target), and chain them, with no demonstrations of that specific list and no retraining. We did not implement or test this, and how well learned policies transfer to unseen targets and still avoid norm violations is an open question.
- The approximate transition model ignores obstacle collisions. A model that accounts for obstacles, possibly using the real environment, should close much of the gap under noise. We did experiment with having the agent actually perform the action in the environment and observe the resulting state, but this makes training significantly slower - executing the action, even with the headless simulation, is slower than a simple `x -> x + 0.25` operation, and would have made training times several orders of magnitude slower. Thus, we elected to use the approximate transition model as it was not the focus of the project. An intermediate approach could take into accound the coordinate locations of all obstacles and only updating coordinates when not bumping into an obstacle.

## Running it

Install the extra requirements (`pip install -r requirements.txt`), then:

```bash
# Train the HIRL pipeline on a sample list and plot a sampled trajectory
# (omit --learn_mode to use our pre-trained parameters)
python irl_agents_separate.py --learn_mode

# Execute the generated trajectory in the supermarket environment (two terminals)
python socket_env.py --player_speed=0.25 --file=start-state.txt
python run-generated-irl-trajectory.py
```

## Acknowledgments

Thank you to the course instructors for the Proper Shopper environment and for the project framework.
