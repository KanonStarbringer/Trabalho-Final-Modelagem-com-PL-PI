import pandas as pd
import streamlit as st
from pyomo.environ import *
import matplotlib.pyplot as plt

# Set page title and favicon
st.set_page_config(page_title="Online 1-machine scheduler", page_icon="https://presearch.com/images?q=UFF#view")

# Function to solve the scheduling problem
def solve_scheduling_problem(data):
    r = data.iloc[0, 1:].astype(int).tolist()  # Tempo de liberação
    d = data.iloc[1, 1:].astype(int).tolist()  # Duração do job
    p = data.iloc[2, 1:].astype(int).tolist()  # Prazo do job
    w = data.iloc[3, 1:].astype(int).tolist()  # Penalidade por atraso
    tipo = data.iloc[4, 1:].tolist()  # Tipo do job
    n = len(r)

    tmax = max(p) + max(d) + 10  # Buffer para acomodar atrasos

    model = ConcreteModel()

    model.J = RangeSet(n)
    model.T = RangeSet(0, tmax)

    model.r = Param(model.J, initialize={i+1: r[i] for i in range(n)})
    model.d = Param(model.J, initialize={i+1: d[i] for i in range(n)})
    model.p = Param(model.J, initialize={i+1: p[i] for i in range(n)})
    model.w = Param(model.J, initialize={i+1: w[i] for i in range(n)})
    model.tipo = Param(model.J, initialize={i+1: tipo[i] for i in range(n)}, within=Any)

    model.y = Var(((j, t) for j in model.J for t in model.T if t >= r[j-1]), domain=Binary)

    def objective_function(model):
        penalty = 0
        for j in model.J:
            for t in model.T:
                if (j, t) not in model.y.index_set():
                    continue
                if t >= model.p[j] - model.d[j] + 1:
                    if t <= model.p[j] + 5 - model.d[j]:
                        penalty += model.w[j] * (t - (model.p[j] - model.d[j])) * model.y[j, t]
                    else:
                        penalty += (model.w[j] * 5 + 2 * model.w[j] * (t - (model.p[j] + 5 - model.d[j]))) * model.y[j, t]
        return penalty

    model.obj = Objective(rule=objective_function, sense=minimize)

    def start_constraint(model, j):
        return sum(model.y[j, t] for t in model.T if t >= model.r[j] and t <= tmax - model.d[j]) == 1
    model.start_constraint = Constraint(model.J, rule=start_constraint)

    def execution_constraint(model, t):
        return sum(model.y[j, s] for j in model.J for s in range(max(0, t - model.d[j] + 1), t + 1) if (j, s) in model.y) <= 1
    model.execution_constraint = Constraint(model.T, rule=execution_constraint)

    def setup_time_constraint(model, j1, j2, t):
        if model.tipo[j1] != model.tipo[j2] and t + model.d[j1] <= tmax:
            if (j1, t) in model.y.index_set() and (j2, t + model.d[j1]) in model.y.index_set():
                return model.y[j1, t] + model.y[j2, t + model.d[j1]] <= 1
        return Constraint.Skip

    model.setup_time_constraint = Constraint(model.J, model.J, model.T, rule=setup_time_constraint)

    solver = SolverFactory('glpk')
    results = solver.solve(model, tee=True)

    schedule = []
    for j in model.J:
        for t in model.T:
            if (j, t) in model.y and model.y[j, t].value == 1:
                schedule.append((j, t))
                break

    total_penalty = value(model.obj)
    return schedule, total_penalty

# Main page
def main():
    st.title("Job Scheduling Optimization")
    st.write("""
    This app performs job scheduling optimization. 
    Upload an Excel file with the job data, and the app will calculate the optimal schedule and display the Gantt chart.
    """)

    st.markdown("""
    **Abstract**:
    This app presents an exact algorithm for the identical parallel machine scheduling problem over a formulation where each variable is indexed by a pair of jobs and a completion time. We show that such a formulation can be handled, in spite of its huge number of variables, through a branch cut and price algorithm enhanced by a number of practical techniques, including a dynamic programming procedure to fix variables by Lagrangean bounds and dual stabilization. The resulting method permits the solution of many instances of the $P||∑w_jT_j$ problem with up to 100 jobs, and having 1 machine. This is the first time that medium-sized instances of the $P||∑w_jT_j$ have been solved to optimality.
    """)


    st.markdown("""
    For more details, please refer to the original [Pessoa et al. 2010](https://www.researchgate.net/profile/Rosiane-De-Freitas-Rodrigues/publication/226907792_Exact_algorithm_over_an_arc-time-indexed_formulation_for_parallel_machine_scheduling_problems/links/0c9605228efb1d404a000000/Exact-algorithm-over-an-arc-time-indexed-formulation-for-parallel-machine-scheduling-problems.pdf?_sg%5B0%5D=started_experiment_milestone&origin=journalDetail&_rtd=e30%3D).
    """)

     # Provide download link for template file
    with open("template.xlsx", "rb") as file:
        btn = st.download_button(
            label="Download Excel Template",
            data=file,
            file_name="template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")

    if uploaded_file:
        data = pd.read_excel(uploaded_file)
        schedule, total_penalty = solve_scheduling_problem(data)
        
        st.write(f"## Total Penalty: {total_penalty}")

        fig, gnt = plt.subplots(figsize=(15, 6))

        gnt.set_xlabel('Time')
        gnt.set_ylabel('Jobs')

        tmax = max(data.iloc[2, 1:].astype(int).tolist()) + max(data.iloc[1, 1:].astype(int).tolist()) + 10

        gnt.set_xlim(0, tmax)
        gnt.set_ylim(0, len(schedule) + 1)

        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
        for (job, start) in schedule:
            job_duration = int(data.iloc[1, job])  # Ensure this is an integer
            st.write(f"Job: {job}, Start: {start}, Duration: {job_duration}")  # Debug statement
            gnt.broken_barh([(start, job_duration)], (job - 0.4, 0.8), facecolors=(colors[job % len(colors)]))

        for (job, start) in schedule:
            job_duration = int(data.iloc[1, job])  # Ensure this is an integer
            gnt.text(start + job_duration / 2, job, f'Job {job}', ha='center', va='center', color='black')

        gnt.set_xticks(range(tmax + 1))
        gnt.grid(True)

        st.pyplot(fig)

if __name__ == "__main__":
    main()
