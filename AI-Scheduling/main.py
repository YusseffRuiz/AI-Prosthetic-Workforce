"""
Program prepared for the simulation approach of paper:
AI-Driven Task Allocation System for Prosthetic Users
AI-based task allocation system that assigns manufacturing tasks dynamically based on prosthetic user constraints
(e.g., fatigue levels, mobility limitations, ergonomic needs)

We will build a rule-based AI system enhanced with reinforcement learning that assigns tasks dynamically based on
prosthetic user constraints. The system will:

✅ Model different manufacturing tasks with varying physical demands.
✅ Simulate worker states (fatigue levels, movement constraints, task efficiency).
✅ Use AI to allocate tasks optimally based on ergonomic constraints and performance metrics.
✅ Evaluate AI-driven task allocation against a traditional, non-adaptive method.

Task Name          |	Physical Demand |   Prosthetic User Constraint
Assembly Work      |		Medium      |	Requires dexterity & light lifting
Machine Operation  |		Low         |	Stationary, requires button pressing & foot pedal
Quality Inspection |		Low	        |	Visual & minimal hand movement
Material Handling  |		High        |	Heavy lifting & mobility-intensive
Packaging          |		Medium      |	Hand movement & coordination


in each area a random distribution of workers, having workers with mobility disabilities and full-abled workers,
where the ones with disabilities is always less than 30%. Amount of people of each area should have from 10 to 20 workers.

The algorithm must be able to perform the scheduling across time.
Simulation should last for 8 hours of work of a day.

The results should be able to be seen graphically, in a map where it shows amount of people in the different sections
and how they move at the different times.
Printing then the time, and amount of people in each location,
and if there is any kind of alerts as well.

"""
import os

import numpy as np
import random
import matplotlib.pyplot as plt

tasks = {
    "Assembly Work": {"physical_demand": 0.6, "dexterity_required": True, "mobility_required": False},
    "Machine Operation": {"physical_demand": 0.3, "dexterity_required": False, "mobility_required": False},
    "Quality Inspection": {"physical_demand": 0.2, "dexterity_required": True, "mobility_required": False},
    "Material Handling": {"physical_demand": 0.9, "dexterity_required": False, "mobility_required": True},
    "Packaging": {"physical_demand": 0.5, "dexterity_required": True, "mobility_required": False}
}


import numpy as np
from WorkEnvironment import WorkEnvironment, RLTaskScheduler, AIWorkEnvironment, RuleBasedWorkEnvironment, RandomWorkEnvironment

save_folder  = "scheduling_images/"


def initialize_workers(work_env):
    # Randomly distribute workers into tasks & store initial values
    min_non_physical_workers = {}

    # Randomly distribute workers into tasks
    for task_name in work_env.tasks.keys():
        if task_name not in ["Machine Operation", "Quality Inspection"]:
            num_workers = np.random.randint(16, 21)  # Random workers per task
        else:
            num_workers = np.random.randint(10, 15)  # Random workers per task
            min_non_physical_workers[task_name] = num_workers  # Save the count
        num_disabled = int(num_workers * np.random.uniform(0.1, 0.3))  # 10-30% disabled workers
        num_full_abled = num_workers - num_disabled  # Remaining are full-abled

        work_env.add_workers(task_name, num_full_abled, num_disabled)


    return work_env, min_non_physical_workers

def run_simulations(environment_class, label, hours=8):
    """ Runs a simulation and stores results for analysis """
    # print(f"\n🔹 **Running {label} Workforce Simulation** 🔹")
    env = environment_class()

    # Store minimum required workers for non-physical tasks
    env, min_non_physical_workers = initialize_workers(env)

    # Pass the minimum workforce to the AIWorkEnvironment
    env.set_min_non_physical_workers(min_non_physical_workers)

    # Run simulation
    env.run_simulation(hours=hours)

    # Store results
    return {
        "label": label,
        "alarm_history": env.alarm_history,
        "work_history": env.work_history
    }


def compute_efficiency(simulation_results, total_work_assigned=3000):
    """ Computes efficiency values for each scheduling method """

    efficiency_scores = {}

    for result in simulation_results:
        label = result["label"]

        # Work Completion Efficiency
        total_work_remaining = sum(result["work_history"][task][-1] for task in result["work_history"].keys())
        work_completed = total_work_assigned - total_work_remaining
        work_efficiency = (work_completed / total_work_assigned) * 100

        # Fatigue Efficiency (Hours without Red or Injury Alarms)
        total_hours = len(next(iter(result["alarm_history"].values()))["yellow"])  # Number of hours simulated
        non_critical_hours = sum(
            all(result["alarm_history"][task]["red"][h] == 0 and result["alarm_history"][task]["injury"][h] == 0
                for task in result["alarm_history"].keys())
            for h in range(total_hours)
        )
        non_warning_hours = sum(
            all(result["alarm_history"][task]["red"][h] == 0 and result["alarm_history"][task]["injury"][h] == 0
                and result["alarm_history"][task]["yellow"][h] == 0
                for task in result["alarm_history"].keys())
            for h in range(total_hours)
        )
        non_fatigue_efficiency = non_warning_hours/total_hours * 100
        fatigue_efficiency = non_critical_hours / total_hours * 100

        # Alarm Severity Score (Weighted Sum)
        total_alarms = sum(
            sum(result["alarm_history"][task]["yellow"]) +
            2 * sum(result["alarm_history"][task]["red"]) +
            3 * sum(result["alarm_history"][task]["injury"])
            for task in result["alarm_history"].keys()
        )

        max_possible_alarms = total_hours * (len(result["alarm_history"]) * 3 +
                                             len(result["alarm_history"]) * 2 +
                                             len(result["alarm_history"]))
        # Maximum possible alarms (all tasks triggering all alarms every hour)

        alarm_efficiency = 100 - (total_alarms / max_possible_alarms * 100)

        # Overall Efficiency Score (OES)
        overall_efficiency = ((work_efficiency + fatigue_efficiency) / 2) - ((total_alarms / max_possible_alarms) * 100)
        if overall_efficiency < 0 :
            overall_efficiency = 0

        efficiency_scores[label] = {
            "Work Efficiency (%)": round(work_efficiency, 2),
            "Fatigue Efficiency (%)": round(fatigue_efficiency, 2),
            "Alarm Severity Score (%)": round(alarm_efficiency,2),
            "Overall Efficiency Score (OES)": round(overall_efficiency, 2),
        }

    return efficiency_scores


def plot_results(simulation_results, pr=False):
    """ Plots alarm history, work completion, and computes efficiency scores """
    hours = range(8)

    # Compute efficiency scores
    efficiency_scores = compute_efficiency(simulation_results)

    # Plot 1: Alarms Over Time
    plt.figure(figsize=(12, 5))
    for result in simulation_results:
        # total_alarms = [sum(result["alarm_history"][task]["yellow"][h] +
        total_alarms = [sum(result["alarm_history"][task]["red"][h] +
                            result["alarm_history"][task]["injury"][h]
                            for task in result["alarm_history"].keys())
                        for h in hours]

        plt.plot(hours, total_alarms, label=result["label"])
        plt.text(hours[-1], total_alarms[-1], f"{total_alarms[-1]:.2f}")

    plt.xlabel("Hour")
    plt.ylabel("Total Alarms Raised")
    plt.title("Total Alarms Raised Per Hour by Scheduling Method")
    plt.legend()
    plt.grid(True)
    image_path = os.path.join(save_folder, "Single_Episode_Alarms_Raised.png")
    plt.savefig(image_path)
    plt.show()


    # Plot 2: Work Completion Over Time
    plt.figure(figsize=(12, 5))
    for result in simulation_results:
        total_remaining_work = [sum(result["work_history"][task][h]
                                    for task in result["work_history"].keys())
                                for h in hours]
        plt.plot(hours, total_remaining_work, label=result["label"])
        plt.text(hours[-1], total_remaining_work[-1], f"{total_remaining_work[-1]:.2f}")

    plt.xlabel("Hour")
    plt.ylabel("Remaining Work Blocks")
    plt.title("Work Completion Over Time by Scheduling Method")
    plt.legend()
    plt.grid(True)
    image_path = os.path.join(save_folder, "Single_Episode_Work_Completion.png")
    plt.savefig(image_path)
    plt.show()


    # Print Efficiency Scores
    if pr:
        print("\n📊 **Efficiency Metrics by Scheduling Method:**")
        for method, scores in efficiency_scores.items():
            print(f"\n🔹 **{method}:**")
            for metric, value in scores.items():
                print(f"   - {metric}: {value}")

    plot_efficiency_metrics(efficiency_scores)


def plot_efficiency_metrics(efficiency_scores, pr=False):
    """ Plots efficiency metrics in a grouped bar chart with correct labels """
    categories = ["Work Efficiency (%)", "Fatigue Efficiency (%)", "Alarm Severity Score (%)", "Overall Efficiency Score (OES)"]
    methods = list(efficiency_scores.keys())

    # Extract data for plotting
    data = {method: [efficiency_scores[method][category] for category in categories] for method in methods}

    # Find the best non-AI values for each metric (Static, Random, Rule-Based)
    non_ai_methods = [method for method in methods if "AI" not in method]
    ai_method = [method for method in methods if "AI" in method][0]

    best_non_ai_values = {}
    for category in categories:
        if category == "Alarm Severity Score":
            # For Alarm Severity, lower is better, so we use min instead of max
            best_non_ai_values[category] = min(efficiency_scores[method][category] for method in non_ai_methods)
        else:
            # For other metrics, higher is better, so we use max
            best_non_ai_values[category] = max(efficiency_scores[method][category] for method in non_ai_methods)

    ai_values = {category: efficiency_scores[ai_method][category] for category in categories}

    # Calculate AI improvement over the best non-AI method
    ai_improvements = {
        category: ((ai_values[category] - best_non_ai_values[category]) / best_non_ai_values[category]) * 100
        if best_non_ai_values[category] != 0 else 0
        for category in categories}


    x = np.arange(len(categories))  # Label locations
    width = 0.2  # Bar width

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, method in enumerate(methods):
        bars = ax.bar(x + (i * width), data[method], width, label=method)
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height, f"{height:.2f}", ha='center', va='bottom', fontsize=10)

    ax.set_xlabel("Efficiency Metrics")
    ax.set_ylabel("Values")
    ax.set_title("Comparison of Workforce Scheduling Efficiency Metrics")
    ax.set_xticks(x + width * (len(methods) / 2))
    ax.set_xticklabels(categories, rotation=15)
    ax.legend()
    ax.grid(axis='y', linestyle="--", alpha=0.7)
    image_path = os.path.join(save_folder, "Single_Episode_EfficiencyMetrics.png")
    plt.savefig(image_path)
    plt.show()


    # Print AI performance comparison against best non-AI method
    if pr:
        print("\n📊 **AI vs Best Non-AI Performance Comparison:**")
        for category, improvement in ai_improvements.items():
            best_method = max(non_ai_methods, key=lambda m: efficiency_scores[m][category])
            print(f"\n🔹 **{category}:**")
            print(f"   - Best Non-AI Method: {best_method} ({best_non_ai_values[category]:.2f})")
            print(f"   - AI Performance: {ai_values[category]:.2f}")
            print(f"   - Improvement Over Best Non-AI: {improvement:.2f}%")


def compute_statistics(results_list):
    """ Computes mean and standard deviation for efficiency metrics across multiple episodes """
    categories = ["Work Efficiency (%)", "Fatigue Efficiency (%)", "Alarm Severity Score (%)",
                  "Overall Efficiency Score (OES)"]
    methods = results_list[0].keys()

    stats = {method: {category: {"mean": 0, "std": 0} for category in categories} for method in methods}

    for method in methods:
        for category in categories:
            values = [result[method][category] for result in results_list]
            stats[method][category]["mean"] = np.mean(values)
            stats[method][category]["std"] = np.std(values)

    return stats


def plot_efficiency_statistical_metrics(stats):
    """ Plots efficiency metrics with mean and standard deviation across multiple episodes """
    categories = ["Work Efficiency (%)", "Fatigue Efficiency (%)", "Alarm Severity Score (%)",
                  "Overall Efficiency Score (OES)"]
    methods = list(stats.keys())

    # Extract data for plotting
    means = {method: [stats[method][category]["mean"] for category in categories] for method in methods}
    stds = {method: [stats[method][category]["std"] for category in categories] for method in methods}

    x = np.arange(len(categories))  # Label locations
    width = 0.2  # Bar width

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, method in enumerate(methods):
        bars = ax.bar(x + (i * width), means[method], width, yerr=stds[method], capsize=5, label=method, alpha=0.7)
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height, f"{height:.2f}", ha='center', va='bottom', fontsize=10)

    ax.set_xlabel("Efficiency Metrics")
    ax.set_ylabel("Values")
    ax.set_title("Comparison of Workforce Scheduling Efficiency Metrics Across 50 Episodes")
    ax.set_xticks(x + width * (len(methods) / 2))
    ax.set_xticklabels(categories, rotation=15)
    ax.legend()
    ax.grid(axis='y', linestyle="--", alpha=0.7)
    image_path = os.path.join(save_folder, "Statistical_EfficiencyMetrics.png")
    plt.savefig(image_path)
    plt.show()


def compare_ai_performance(stats):
    """ Compares AI performance improvement over the best non-AI method using statistical analysis """
    categories = ["Work Efficiency (%)", "Fatigue Efficiency (%)", "Alarm Severity Score (%)",
                  "Overall Efficiency Score (OES)"]
    non_ai_methods = [method for method in stats.keys() if "AI" not in method]
    ai_method = [method for method in stats.keys() if "AI" in method][0]

    best_non_ai_values = {}
    for category in categories:
        if category == "Alarm Severity Score":
            best_non_ai_values[category] = min(stats[method][category]["mean"] for method in non_ai_methods)
        else:
            best_non_ai_values[category] = max(stats[method][category]["mean"] for method in non_ai_methods)

    ai_values = {category: stats[ai_method][category]["mean"] for category in categories}

    ai_improvements = {
        category: ((ai_values[category] - best_non_ai_values[category]) / best_non_ai_values[category]) * 100
        if best_non_ai_values[category] != 0 else 0
        for category in categories}

    # Print AI performance comparison
    print("\n📊 **AI vs Best Non-AI Performance Comparison (Across 50 Episodes):**")
    for category, improvement in ai_improvements.items():
        best_method = min(non_ai_methods,
                          key=lambda m: stats[m][category]["mean"]) if category == "Alarm Severity Score" \
            else max(non_ai_methods, key=lambda m: stats[m][category]["mean"])

        print(f"\n🔹 **{category}:**")
        print(f"   - Best Non-AI Method: {best_method} (Mean: {best_non_ai_values[category]:.2f})")
        print(f"   - AI Performance (Mean): {ai_values[category]:.2f}")
        print(f"   - AI Improvement Over Best Non-AI: {improvement:.2f}%")


def main_test():
    # Create work environment
    work_env = WorkEnvironment()

    work_env, min_non_physical_workers = initialize_workers(work_env)

    # Pass the minimum workforce to the AIWorkEnvironment
    work_env.set_min_non_physical_workers(min_non_physical_workers)

    # Print initial status
    print("\n🏭 **INITIAL WORK ENVIRONMENT STATUS** 🏭")
    work_env.print_status()

    # Run simulation for 8 hours
    print("\n⏳ **STARTING WORK SIMULATION** ⏳")
    work_env.run_simulation(hours=8)

    # Final results and plot
    print("\n✅ **SIMULATION COMPLETED** ✅")
    work_env.print_status()
    work_env.plot_results()  # Generate fatigue & work completion graphs

def main_rl(train=False):
    # Step 1: Train the RL model
    num_tasks = 5
    if train:
        ai_scheduler = RLTaskScheduler(num_tasks)
        ai_scheduler.train(episodes=50000)

    # Step 2: Deploy AI-driven workforce simulation
    print("\n🔹 **Starting AI-Based Workforce Simulation** 🔹")
    work_env = AIWorkEnvironment()

    work_env, min_non_physical_workers = initialize_workers(work_env)

    # Pass the minimum workforce to the AIWorkEnvironment
    work_env.set_min_non_physical_workers(min_non_physical_workers)

    # Print initial status
    print("\n🏭 **INITIAL WORK ENVIRONMENT STATUS** 🏭")
    work_env.print_status()


    # Run AI-powered workforce simulation
    work_env.run_simulation(hours=8)
    # Final results and plot
    print("\n✅ **SIMULATION COMPLETED** ✅")
    work_env.print_status()
    work_env.plot_results()  # Generate fatigue & work completion graphs

def main_compare():
    # Step 2: Run Simulations for All Scheduling Methods
    simulation_results = []

    simulation_results.append(run_simulations(WorkEnvironment, "Static Scheduling"))
    simulation_results.append(run_simulations(RandomWorkEnvironment, "Random Scheduling"))
    simulation_results.append(run_simulations(RuleBasedWorkEnvironment, "Rule-Based Scheduling"))
    simulation_results.append(run_simulations(AIWorkEnvironment, "AI-Based Scheduling"))

    # Step 3: Generate Comparison Plots
    plot_results(simulation_results)

def main_statistical_analysis():
    # Simulate results over 50 episodes (Replace with actual simulation data)
    np.random.seed(42)  # For reproducibility
    num_episodes = 50
    simulated_results = []

    for _ in range(num_episodes):
        simulations_per_day = []
        simulations_per_day.append(run_simulations(WorkEnvironment, "Static Scheduling"))
        simulations_per_day.append(run_simulations(RandomWorkEnvironment, "Random Scheduling"))
        simulations_per_day.append(run_simulations(RuleBasedWorkEnvironment, "Rule-Based Scheduling"))
        simulations_per_day.append(run_simulations(AIWorkEnvironment, "AI-Based Scheduling"))
        simulated_results.append(compute_efficiency(simulations_per_day))

    # Compute statistics
    stats = compute_statistics(simulated_results)

    # Plot results
    plot_efficiency_statistical_metrics(stats)

    # Compare AI vs Best Non-AI performance
    compare_ai_performance(stats)




if __name__ == '__main__':
    # main_test()
    # main_rl(train=False)
    # main_compare()
    main_statistical_analysis()



"""
Results:

**AI vs Best Non-AI Performance Comparison (Across 50 Episodes):**

🔹 **Work Efficiency (%):**
   - Best Non-AI Method: Static Scheduling (Mean: 99.11)
   - AI Performance (Mean): 95.76
   - AI Improvement Over Best Non-AI: -3.38%

🔹 **Fatigue Efficiency (%):**
   - Best Non-AI Method: Rule-Based Scheduling (Mean: 74.50)
   - AI Performance (Mean): 74.25
   - AI Improvement Over Best Non-AI: -0.34%

🔹 **Alarm Severity Score (%):**
   - Best Non-AI Method: Rule-Based Scheduling (Mean: 67.70)
   - AI Performance (Mean): 82.24
   - AI Improvement Over Best Non-AI: 21.48%

🔹 **Overall Efficiency Score (OES):**
   - Best Non-AI Method: Rule-Based Scheduling (Mean: 53.23)
   - AI Performance (Mean): 67.25
   - AI Improvement Over Best Non-AI: 26.32%
"""