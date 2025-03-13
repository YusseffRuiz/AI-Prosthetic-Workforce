# 🦾 AI-Powered Prosthetic Workforce Optimization

### **🚀 Reinforcement Learning for Dynamic Task Allocation in Industrial Environments**
This repository contains the implementation of an **AI-driven task allocation system** designed to optimize **workforce scheduling** for **mobility-impaired workers** in industrial environments. The project integrates **reinforcement learning (Q-learning)** and **Human Digital Twin (HDT) simulation** to enhance **task distribution, fatigue management, and ergonomic efficiency** in the context of **Industry 5.0**.

---

## **📌 Project Overview**
Workers with mobility impairments, such as those using prosthetic limbs, face **increased fatigue and movement inefficiency** in manufacturing roles. This project addresses these challenges by:
- 📌 **Implementing an RL-based scheduler** that dynamically reallocates tasks based on **fatigue levels, efficiency, and workload distribution**.
- 📌 **Simulating biomechanical movement** through a **Human Digital Twin (HDT)** to analyze **energy expenditure and movement inefficiency**.
- 📌 **Comparing AI-based scheduling with traditional methods** (Static, Rule-Based, Random) to evaluate overall efficiency.

---

## **🛠️ Technologies Used**
🔹 **Python** – Core programming language  
🔹 **Reinforcement Learning (Q-Learning)** – AI scheduling framework  
🔹 **NumPy, Matplotlib, Seaborn** – Data processing & visualization  
🔹 **MyoSuite** – Digital Twin simulation of prosthetic worker movement  
🔹 **GitHub Actions** – Continuous development & code updates  

---

## **📊 Experimental Setup**
This project consists of **two major experiments**:

### **1️⃣ AI-Driven Task Allocation System**
- 🏭 **Workforce Simulation:** Workers are assigned tasks based on **fatigue levels** and **energy consumption**.
- 🎯 **AI vs. Traditional Scheduling:** Reinforcement Learning is compared against:
  - **Static Scheduling**
  - **Random Task Assignment**
  - **Rule-Based Scheduling**
- 📈 **Performance Metrics:** Work efficiency, fatigue management, and injury risk minimization.

### **2️⃣ Human Digital Twin (HDT) Analysis**
- 🦿 **Prosthetic Worker Biomechanics:** Evaluates the **energy cost of movement** and **step efficiency**.
- 🔬 **Key Findings:**
  - 🏃 Prosthetic users travel **7.55% less distance** than able-bodied workers.
  - ⚡ They consume **22.17% more energy** for the same workload.
  - 🔄 Require **32.60% more energy per meter walked**.
- 🛠️ **Industry 5.0 Implications:** AI-driven task reallocation can **reduce unnecessary movement** and **optimize workforce distribution**.

---

## **📌 Key Results**
| **Metric** | **Best Non-AI Method** | **AI Performance** | **Improvement** |
|------------|----------------------|-------------------|----------------|
| **Work Efficiency (%)** | 99.11 | 95.76 | 🔽 **-3.38%** |
| **Fatigue Efficiency (%)** | 74.50 | 74.25 | 🔽 **-0.34%** |
| **Alarm Severity Score (%)** | 67.70 | 82.24 | 🔼 **+21.48%** |
| **Overall Efficiency Score (OES)** | 53.23 | 67.25 | 🔼 **+26.32%** |

---

## **📦 Repository Structure**
