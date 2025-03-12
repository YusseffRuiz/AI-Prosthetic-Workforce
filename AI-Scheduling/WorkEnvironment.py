import os

from worker import FullAbledWorker, DisabledWorker
import numpy as np
import matplotlib.pyplot as plt
from Reinforce import RLTaskScheduler
import random


save_folder = "scheduling_images/"

class WorkEnvironment:
    def __init__(self):
        """ Initialize the work environment with tasks and workers """
        self.tasks = {
            "Assembly Work": {"workload": 1000, "workers": [], "physical": True},
            "Material Handling": {"workload": 1000, "workers": [], "physical": True},
            "Packaging": {"workload": 1000, "workers": [], "physical": True},
            "Machine Operation": {"workload": None, "workers": [], "physical": False},
            "Quality Inspection": {"workload": None, "workers": [], "physical": False}
        }
        self.total_workers = 0  # Total workers in the environment
        self.worker_counts = {task: {"abled": 0, "disabled": 0} for task in self.tasks.keys()}  # Store worker numbers
        self.fatigue_history = {task: [] for task in self.tasks.keys()}  # Fatigue tracking
        self.work_history = {task: [] for task in self.tasks.keys() if self.tasks[task]["workload"] is not None}  # Work completion tracking
        # Track alarms per hour & task
        self.alarm_history = {task: {"yellow": [], "red": [], "injury": []} for task in self.tasks.keys()}
        self.workforce_distribution = []  # Store workforce data per hour
        self.min_non_physical_workers = {}  # This will be set dynamically
        self.sim_name = 'Static'


    def add_workers(self, task_name, num_full_abled, num_disabled):
        """ Assign workers to a task with a mix of full-abled and disabled workers """
        for _ in range(num_full_abled):
            self.tasks[task_name]["workers"].append(FullAbledWorker(worker_id=self.total_workers))
            self.total_workers += 1
        for _ in range(num_disabled):
            self.tasks[task_name]["workers"].append(DisabledWorker(worker_id=self.total_workers))
            self.total_workers += 1

        # Store worker distribution
        self.worker_counts[task_name]["abled"] = num_full_abled
        self.worker_counts[task_name]["disabled"] = num_disabled

    def print_worker_distribution(self):
        """ Print the number of full-abled and disabled workers assigned to each task """
        print("\n👷‍♂️ **Worker Distribution Per Task** 👷‍♀️")
        for task, counts in self.worker_counts.items():
            print(f"{task}: {counts['abled']} Abled Workers | {counts['disabled']} Disabled Workers")

    def assign_tasks(self):
        pass

    def set_min_non_physical_workers(self, min_workers):
        """ Dynamically set minimum non-physical workforce from simulation start """
        self.min_non_physical_workers = min_workers

    def simulate_hour(self, current_hour, pr=False):
        """ Simulates one hour of work, updates worker states, and tracks alarms """
        alarm_counts = {task: {"yellow": 0, "red": 0, "injury": 0} for task in self.tasks.keys()}

        # **AI dynamically assigns tasks before each hour begins**
        self.assign_tasks()

        # Track workforce distribution per task
        workforce_counts = {task: {"abled": 0, "disabled": 0} for task in self.tasks.keys()}

        for task_name, task_info in self.tasks.items():
            workers = task_info["workers"]
            total_work_done = 0  # Track work progress
            fatigue_levels = []

            # Track number of full-abled and disabled workers
            abled_count = sum(1 for worker in workers if isinstance(worker, FullAbledWorker))
            disabled_count = sum(1 for worker in workers if isinstance(worker, DisabledWorker))

            workforce_counts[task_name] = {"abled": abled_count, "disabled": disabled_count}

            for worker in workers:
                total_work_done += worker.work()  # Workers complete work
                worker.update_fatigue(task_type=task_name)  # Fatigue increases
                fatigue_levels.append(worker.energy)  # Track energy levels

                # Check for alarms
                if worker.energy < 0.50 and worker.energy >= 0.30:
                    alarm_counts[task_name]["yellow"] += 1
                elif worker.energy < 0.30 and worker.energy > 0.0:
                    alarm_counts[task_name]["red"] += 1
                elif worker.energy == 0.0:
                    alarm_counts[task_name]["injury"] += 1

            # Reduce remaining workload (for finite work tasks)
            if task_info["workload"] is not None:
                task_info["workload"] = max(0, task_info["workload"] - total_work_done)
                self.work_history[task_name].append(task_info["workload"])  # Save work progress

            # Save fatigue level for this hour
            self.fatigue_history[task_name].append(np.mean(fatigue_levels) if fatigue_levels else 0)

        # Store workforce distribution per hour
        self.workforce_distribution.append(workforce_counts)


        # Store alarm history per hour
        for task in self.tasks.keys():
            self.alarm_history[task]["yellow"].append(alarm_counts[task]["yellow"])
            self.alarm_history[task]["red"].append(alarm_counts[task]["red"])
            self.alarm_history[task]["injury"].append(alarm_counts[task]["injury"])

        # Print alerts for this hour
        if pr:
            if any(sum(alarm_counts[task].values()) > 0 for task in self.tasks.keys()):
                print(f"\n🚨 **ALERTS TRIGGERED IN HOUR {current_hour}:** 🚨")
                for task, counts in alarm_counts.items():
                    if counts["yellow"] > 0:
                        print(f"⚠️ {counts['yellow']} Yellow Warnings in {task}")
                    if counts["red"] > 0:
                        print(f"🔴 {counts['red']} Critical Warnings in {task}")
                    if counts["injury"] > 0:
                        print(f"🛑 {counts['injury']} Injury Alerts in {task}")

    def print_status(self):
        """ Displays the current status of work progress and worker fatigue levels """
        print("\n🔹 **Current Work Progress & Worker States:**")
        for task_name, task_info in self.tasks.items():
            workload = task_info["workload"]
            print(f"{task_name}: Work Remaining - {'Constant Work' if workload is None else workload} blocks")
            fatigues = [worker.energy for worker in task_info["workers"]]
            avg_fatigue = np.mean(fatigues) if fatigues else 0
            print(f"   - Avg Fatigue: {avg_fatigue:.2f}")

    def run_simulation(self, hours=8):
        """ Runs the work simulation for the specified number of hours """
        # self.print_worker_distribution()  # Print worker counts before starting
        self.workforce_distribution.append(self.worker_counts)
        for hour in range(hours):
            # print(f"\n⏳ **Hour {hour + 1}**")
            self.simulate_hour(current_hour=hour + 1)
            # self.print_status()
            # self.print_worker_distribution()
        # print("\n✅ **Simulation Completed!** ✅")
        # self.print_worker_distribution()  # Print worker counts after simulation
        # self.plot_alarm_history()  # Generate alarm history graph
        # self.plot_workforce_distribution()  # **New graph**

    def plot_results(self):
        """ Generates fatigue and work progress graphs """
        plt.figure(figsize=(12, 5))

        # Plot fatigue levels
        plt.subplot(1, 2, 1)
        for task, fatigue_data in self.fatigue_history.items():
            plt.plot(range(len(fatigue_data)), fatigue_data, label=task)
        plt.xlabel("Hour")
        plt.ylabel("Average Energy Left")
        plt.title(f"Fatigue Progression Over Time for {self.sim_name} Assignment")
        plt.legend()

        # Plot work completion
        plt.subplot(1, 2, 2)
        for task, work_data in self.work_history.items():
            plt.plot(range(len(work_data)), work_data, label=task)
        plt.xlabel("Hour")
        plt.ylabel("Remaining Work Blocks")
        plt.title(f"Work Completion Over Time for {self.sim_name} Assignment")
        plt.legend()

        plt.tight_layout()
        plt.show()

    def plot_alarm_history(self):
        """ Generates a bar chart of alarms per hour per task """
        plt.figure(figsize=(12, 6))

        hours = list(range(len(next(iter(self.alarm_history.values()))["yellow"])))

        for task, alarms in self.alarm_history.items():
            plt.plot(hours, alarms["yellow"], marker="o", linestyle="dashed", label=f"⚠️ Yellow ({task})")
            plt.plot(hours, alarms["red"], marker="s", linestyle="dashed", label=f"🔴 Red ({task})")
            plt.plot(hours, alarms["injury"], marker="x", linestyle="dashed", label=f"🛑 Injury ({task})")

        plt.xlabel("Hour")
        plt.ylabel("Number of Alarms")
        plt.title(f"Alarms Triggered Per Hour for {self.sim_name} Assignment")
        plt.legend()
        plt.grid(True)
        image_path = os.path.join(save_folder, f"Alarms_Triggered_{self.sim_name}.png")
        plt.savefig(image_path)
        plt.show()

    def plot_workforce_distribution(self):
        """ Generates a stacked bar chart to show workforce distribution per hour """
        hours = len(self.workforce_distribution)
        task_names = list(self.tasks.keys())

        fig, axes = plt.subplots(3, 3, figsize=(16, 12))  # 3 rows, 3 columns (for 9-hour simulation)
        fig.suptitle("Workforce Distribution Per Task Over Time", fontsize=16)

        for hour in range(hours):
            ax = axes[hour // 3, hour % 3]  # Position in subplot grid

            if hour == 0:
                workforce_counts = self.workforce_distribution[0]  # Use initial state for Hour 0
                title = "Hour 0 (Initial State)"
            else:
                workforce_counts = self.workforce_distribution[hour - 1]  # Use actual hour data
                title = f"Hour {hour}"

            # Extract full-abled and disabled worker counts separately
            full_abled_counts = [workforce_counts[task]["abled"] for task in task_names]
            disabled_counts = [workforce_counts[task]["disabled"] for task in task_names]

            x = np.arange(len(task_names))  # X-axis positions

            # Bar chart with stacked full-abled and disabled worker counts
            ax.bar(x, full_abled_counts, label="Full Abled", color="blue", alpha=0.7)
            ax.bar(x, disabled_counts, label="Disabled", color="red", alpha=0.7, bottom=full_abled_counts)

            ax.set_xticks(x)
            ax.set_xticklabels(task_names, rotation=45, ha="right")
            ax.set_ylabel("Workers")
            ax.set_title(title)
            ax.legend()

        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit titles
        plt.show()



class AIWorkEnvironment(WorkEnvironment):
    def __init__(self):
        super().__init__()
        num_tasks = len(self.tasks)
        self.rl_scheduler = RLTaskScheduler(num_tasks)
        self.rl_scheduler.load_trained_model()  # Load RL model weights
        self.sim_name = "AI"

    def assign_tasks(self):
        """ AI dynamically assigns workers to tasks while ensuring workforce stability """
        new_worker_counts = {task: {"abled": 0, "disabled": 0} for task in self.tasks.keys()}

        for task_name, task_info in self.tasks.items():
            for worker in task_info["workers"]:
                fatigue_level = worker.energy
                biomechanical_state = worker.efficiency
                workers_remaining = sum(len(self.tasks[task]["workers"]) for task in ["Machine Operation", "Quality Inspection"])

                assigned_task_idx = self.rl_scheduler.allocate_task(fatigue_level, biomechanical_state, workers_remaining)
                assigned_task = list(self.tasks.keys())[assigned_task_idx]

                # Ensure minimum workforce in non-physical tasks
                if task_name in self.min_non_physical_workers:
                    required_min = self.min_non_physical_workers[task_name]
                    if workers_remaining <= required_min:
                        continue  # Skip reassignment to maintain stability

                # Prioritize moving fatigued workers to non-physical tasks
                if fatigue_level < 0.6 and assigned_task not in ["Machine Operation", "Quality Inspection"]:
                    assigned_task = np.random.choice(["Machine Operation", "Quality Inspection"])

                # If worker is reassigned, update workforce tracking
                if worker.current_task != assigned_task:
                    self.tasks[task_name]["workers"].remove(worker)  # Remove from old task
                    self.tasks[assigned_task]["workers"].append(worker)  # Add to new task
                    worker.current_task = assigned_task

                # Count workers in each task
                if isinstance(worker, FullAbledWorker):
                    new_worker_counts[assigned_task]["abled"] += 1
                else:
                    new_worker_counts[assigned_task]["disabled"] += 1

        # Update workforce tracking with new assignments
        self.worker_counts = new_worker_counts


class RuleBasedWorkEnvironment(WorkEnvironment):
    def __init__(self):
        """ Initializes a rule-based scheduling environment """
        super().__init__()
        self.sim_name="Rule-Based"

    def assign_tasks(self):
        """ Reassigns workers based on simple fatigue-based rules instead of AI """
        new_worker_counts = {task: {"abled": 0, "disabled": 0} for task in self.tasks.keys()}

        for task_name, task_info in self.tasks.items():
            for worker in task_info["workers"]:
                fatigue_level = worker.energy

                # If fatigue goes below 50%, reassign to a non-physical task to avoid injuries
                if fatigue_level < 0.50:
                    assigned_task = np.random.choice(["Machine Operation", "Quality Inspection"])

                # If fatigue is high (>75%), move worker back to a physical task (if possible)
                elif fatigue_level > 0.75:
                    assigned_task = np.random.choice(["Assembly Work", "Material Handling", "Packaging"])

                # Otherwise, keep the worker in the same task
                else:
                    assigned_task = task_name

                    # Ensure minimum workforce in non-physical areas
                if task_name in ["Machine Operation", "Quality Inspection"] and len(self.tasks[task_name]["workers"]) <= \
                        self.min_non_physical_workers[task_name]:
                    assigned_task = task_name  # Do not allow movement

                # If worker is reassigned, update workforce tracking
                if worker.current_task != assigned_task:
                    self.tasks[task_name]["workers"].remove(worker)
                    self.tasks[assigned_task]["workers"].append(worker)
                    worker.current_task = assigned_task

                # Count workers in each task
                if isinstance(worker, FullAbledWorker):
                    new_worker_counts[assigned_task]["abled"] += 1
                else:
                    new_worker_counts[assigned_task]["disabled"] += 1

        # Update worker tracking
        self.worker_counts = new_worker_counts


class RandomWorkEnvironment(WorkEnvironment):
    def __init__(self):
        """ Initializes a random scheduling environment """
        super().__init__()
        self.sim_name = "Random"

    def random_reassign_tasks(self):
        """ Reassigns workers randomly every hour while maintaining non-physical workforce constraints """
        new_worker_counts = {task: {"abled": 0, "disabled": 0} for task in self.tasks.keys()}

        for task_name, task_info in self.tasks.items():
            for worker in task_info["workers"]:

                # Randomly decide if worker will be reassigned (50% chance)
                if random.random() < 0.3:
                    assigned_task = random.choice(list(self.tasks.keys()))
                else:
                    assigned_task = task_name  # Keep in the same task

                # Ensure minimum workforce in non-physical areas
                if task_name in ["Machine Operation", "Quality Inspection"]:
                    required_min = self.min_non_physical_workers[task_name]
                    if len(self.tasks[task_name]["workers"]) <= required_min:
                        assigned_task = task_name  # Prevent movement

                # If worker is reassigned, update workforce tracking
                if worker.current_task != assigned_task:
                    self.tasks[task_name]["workers"].remove(worker)
                    self.tasks[assigned_task]["workers"].append(worker)
                    worker.current_task = assigned_task

                # Count workers in each task
                if isinstance(worker, FullAbledWorker):
                    new_worker_counts[assigned_task]["abled"] += 1
                else:
                    new_worker_counts[assigned_task]["disabled"] += 1

        # Update worker tracking
        self.worker_counts = new_worker_counts