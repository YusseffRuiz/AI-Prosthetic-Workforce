import numpy as np
import pickle


import numpy as np
import pickle
import random

class RLTaskScheduler:
    def __init__(self, num_tasks, alpha=1e-4, gamma=0.98, epsilon=0.2):
        self.num_tasks = num_tasks
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration-exploitation tradeoff
        self.q_table = {}  # Dictionary-based Q-table for flexible workforce sizes

    def get_q_values(self, fatigue_level, biomechanical_state, workers_remaining):
        """ Retrieve Q-values for a given state, initialize if not found """
        state = (round(fatigue_level, 1), biomechanical_state, workers_remaining)
        if state not in self.q_table:
            self.q_table[state] = np.random.uniform(-1, 1, self.num_tasks)  # Initialize Q-values
        return self.q_table[state]

    def train(self, episodes=1000, num_workers=100):
        """ Train the RL model to optimize workforce scheduling dynamically """
        total_work = 1000
        for _ in range(episodes):
            # Randomly set the daily minimum number of workers in non-physical areas
            min_non_physical_workers = np.random.randint(5, 15)  # Varies per day
            work_remaining = total_work  # Total work to be completed in the day
            for _ in range(num_workers):
                fatigue_level = np.random.uniform(0.8, 1)  # Start with random energy
                biomechanical_state = np.random.choice(["optimal", "normal", "decayed"])

                workers_remaining = np.random.randint(min_non_physical_workers, num_workers)

                # Choose action (task assignment) using epsilon-greedy policy
                q_values = self.get_q_values(fatigue_level, biomechanical_state, workers_remaining)
                if np.random.uniform(0, 1) < self.epsilon:
                    action = np.random.randint(0, self.num_tasks)  # Explore
                else:
                    action = np.argmax(q_values)  # Exploit best action

                # Fatigue impact per task
                if action in [0, 1, 2]:  # Physical Tasks
                    fatigue_penalty = np.random.uniform(0, 0.8)  # Increases fatigue
                else:  # Non-physical Tasks (Recovery)
                    fatigue_penalty = -0.15  # Decreases fatigue (Energy recovery)

                # Modify Reward Function:
                fatigue_rw = 0
                remaining_rw = 0
                fatigue_level -= fatigue_penalty

                if workers_remaining < min_non_physical_workers:
                    remaining_rw = -5  # Penalize dropping below daily minimum workers in non-physical areas

                if fatigue_penalty < 0:
                    fatigue_rw += 3

                if 0.6 < fatigue_level < 0.8:
                    fatigue_rw +=3
                elif fatigue_level > 0.8:
                    fatigue_rw += 5  # Reward resting fatigued workers
                else:
                    fatigue_rw += 0

                # **Work Efficiency Reward:**
                work_completed = workers_remaining * np.random.randint(4, 10)
                work_remaining -= work_completed
                work_efficiency_reward = max(0, (1 - (work_remaining / total_work)) * 10)  # Scaled reward

                # Final reward function:
                reward = fatigue_rw + remaining_rw + work_efficiency_reward - (
                        0.2 if action != np.argmax(q_values) else 0)

                # Update Q-table using Bellman equation
                q_values[action] += self.alpha * (
                    reward + self.gamma * np.max(q_values) - q_values[action]
                )

        # Save trained model
        with open("rl_task_scheduler.pkl", "wb") as f:
            pickle.dump(self.q_table, f)
        print("✅ Model trained and weights saved successfully!")

    def allocate_task(self, fatigue_level, biomechanical_state, workers_remaining):
        """ Assigns task dynamically based on worker's fatigue level and biomechanical condition """
        q_values = self.get_q_values(fatigue_level, biomechanical_state, workers_remaining)
        return np.argmax(q_values)  # Otherwise, choose the best action



    def load_trained_model(self):
        """ Load trained RL model weights """
        try:
            with open("rl_task_scheduler.pkl", "rb") as f:
                self.q_table = pickle.load(f)
            print("✅ Pre-trained model loaded successfully!")
        except FileNotFoundError:
            print("⚠️ No trained model found. Run training first.")

