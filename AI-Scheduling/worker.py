import numpy as np


# Worker class for general properties
class Worker:
    def __init__(self, worker_id, efficiency="optimal"):
        self.worker_id = worker_id
        self.energy = np.random.uniform(0.8, 1.0)  # Initial energy (1.0 = fully rested)
        self.efficiency = efficiency  # "optimal", "normal", or "decayed"
        self.current_task = None  # Track worker’s assigned task
        self.fatigue_constant = 0.1  # Default k value for fatigue function
        self.continuous_hours = 0

    @staticmethod
    def fatigue_rate(time_hours):
        # Fatigue levels, including risk of injuries and efficiency can decrease less than 10-13% when a week of work
        # is of 40 hours. DOI:10.1061/(ASCE)0733-9364(1997)123:2(181), 10.1002/ajim.20307
        # 10.1061/(ASCE)0733-9364(2005)131:6(734)

        return np.random.uniform(0.01, 0.13)

    def work(self):
        """ Determines how much work a worker can complete in one hour. """
        if self.efficiency == "optimal":
            return 10  # 10 blocks per hour
        elif self.efficiency == "normal":
            return 7  # 7 blocks per hour
        else:
            return 4  # 4 blocks per hour (decayed efficiency)

    def update_fatigue(self, task_type):
        """ Updates worker's energy level and efficiency based on fatigue. """
        """ Updates worker's energy level based on their assigned task. """
        if task_type in ["Machine Operation", "Quality Inspection"]:
            self.energy = min(1.0, self.energy + 0.15)  # **Recover 15% energy per hour**
            self.continuous_hours = 0
        else:
            self.continuous_hours += 1
            self.energy = self.fatigue_rate(self.continuous_hours)  # **Physical roles reduce energy**

        if self.energy < 0.5:
            self.efficiency = "decayed"
        elif self.energy < 0.7:
            self.efficiency = "normal"
        else:
            self.efficiency = "optimal"

# Full-abled worker class (inherits from Worker)
class FullAbledWorker(Worker):
    def __init__(self, worker_id):
        super().__init__(worker_id, efficiency="optimal")

    def fatigue_rate(self, time_hours):
        # Based on Fatigue Function levels by: DOI: 10.1177/074873099129000920 and PMID: 15018265
        # Computes energy depletion using H_0 * e^(-kt)
        return self.energy * np.exp(-self.fatigue_constant * time_hours/2)


# Disabled worker class (inherits from Worker)
class DisabledWorker(Worker):
    def __init__(self, worker_id):
        super().__init__(worker_id, efficiency="normal")  # Starts as "normal" efficiency

    def fatigue_rate(self, time_hours):
        # Fatigue levels are higher for Workers with mobility issues
        return self.energy * (np.exp(-self.fatigue_constant * time_hours/2) + 0.3*np.exp(-self.fatigue_constant * time_hours/2))