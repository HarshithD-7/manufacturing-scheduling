import numpy as np
import pandas as pd
import pygad
import matplotlib.pyplot as plt
import seaborn as sns
import random

from datetime import datetime, timedelta
import matplotlib.dates as mdates
import matplotlib.patches as patches

# Constants for shift
SHIFT_START = datetime.strptime("09:00", "%H:%M").replace(year=2025, month=5, day=14)
SHIFT_END = datetime.strptime("17:30", "%H:%M").replace(year=2025, month=5, day=14)

machines = {
    "VMC 1": {"operation": "Milling", "available_from": SHIFT_START},
    "VMC 2": {"operation": "Milling", "available_from": SHIFT_START},
    "Turning Center 1": {"operation": "Turning", "available_from": SHIFT_START},
    "Turning Center 2": {"operation": "Turning", "available_from": SHIFT_START},
    "Turn Mill": {"operation": "Turning", "available_from": SHIFT_START},
    "5 Axis Milling": {"operation": "5-Axis Machining", "available_from": SHIFT_START},
    "Surface Grinding": {"operation": "Surface Grinding", "available_from": SHIFT_START},
    "ID Grinder": {"operation": "ID Grinding", "available_from": SHIFT_START},
    "OD Grinder": {"operation": "OD Grinding", "available_from": SHIFT_START},
    "Cutter(Band Saw)": {"operation": "Cutter Operation", "available_from": SHIFT_START}
}

# Static colors for machine types to ensure consistency in the Gantt chart
MACHINE_COLORS = {
    "VMC 1": "skyblue",
    "VMC 2": "deepskyblue",
    "Turning Center 1": "lightgreen",
    "Turning Center 2": "mediumseagreen",
    "Turn Mill": "limegreen",
    "5 Axis Milling": "lightcoral",
    "Surface Grinding": "salmon",
    "ID Grinder": "plum",
    "OD Grinder": "orchid",
    "Cutter(Band Saw)": "orange"
}

# Component colors for differentiation in the Gantt chart - using darker shades
COMPONENT_COLORS = {}

# Global variables for GA optimization
all_components = []
ga_instance = None
best_solution = None

def is_within_shift(time):
    """Check if a time is within the working shift."""
    curr_shift_start = datetime.combine(time.date(), SHIFT_START.time())
    curr_shift_end = datetime.combine(time.date(), SHIFT_END.time())
    return curr_shift_start <= time <= curr_shift_end

def next_working_time(current_time):
    """Get the next valid working time."""
    if is_within_shift(current_time):
        return current_time
    else:
        # If after shift end, go to next day's shift start
        if current_time.time() > SHIFT_END.time():
            next_day = current_time + timedelta(days=1)
            return datetime.combine(next_day.date(), SHIFT_START.time())
        # If before shift start, go to today's shift start
        elif current_time.time() < SHIFT_START.time():
            return datetime.combine(current_time.date(), SHIFT_START.time())

def parse_date(date_str):
    """Parse date string into datetime object."""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        print(f"Invalid date format: {date_str}. Please use YYYY-MM-DD.")
        return None

def get_user_inputs():
    """Get user inputs for multiple components."""
    print("=== Multi-Component Scheduler with Genetic Algorithm ===")

    components = []
    while True:
        print("\n--- New Component Entry ---")
        component_name = input("Enter component name (or 'done' to finish): ")
        if component_name.lower() == 'done':
            if not components:
                print("Please enter at least one component.")
                continue
            break

        quantity = int(input(f"Enter quantity to produce for {component_name}: "))
        due_date_str = input(f"Enter due date for {component_name} (YYYY-MM-DD): ")
        due_date = parse_date(due_date_str)
        if not due_date:
            continue

        priority = int(input(f"Enter priority for {component_name} (1-10, 10 being highest): "))

        print("\nAvailable machines:")
        for idx, m in enumerate(machines):
            print(f"{idx + 1}. {m} - {machines[m]['operation']}")

        operations = []
        while True:
            choice = input("Select machine by number for operation order (Enter 'done' when finished): ")
            if choice.lower() == 'done':
                if not operations:
                    print("Please select at least one machine.")
                    continue
                break
            try:
                idx = int(choice) - 1
                machine_name = list(machines.keys())[idx]
                if machine_name not in operations:
                    operations.append(machine_name)
            except (IndexError, ValueError):
                print("Invalid selection. Try again.")

        cycle_times = {}
        setup_times = {}
        for op in operations:
            cycle_times[op] = float(input(f"Enter cycle time (minutes) for {op}: "))
            setup_times[op] = float(input(f"Enter setup time (minutes) for {op}: "))

        components.append({
            "name": component_name,
            "quantity": quantity,
            "due_date": due_date,
            "priority": priority,
            "operations": operations,  # This already stores operations in the order they should be executed
            "cycle_times": cycle_times,
            "setup_times": setup_times,
        })

    return components

def evaluate_schedule(schedule, components):
    """Evaluate the quality of a schedule based on multiple objectives."""
    # Initialize metrics
    tardiness = 0
    makespan = 0
    machine_idle_time = 0
    priority_weighted_completion = 0

    # Get completion times for each component
    completion_times = {}
    for task in schedule:
        comp_name = task["Component"]
        if comp_name not in completion_times or task["End"] > completion_times[comp_name]:
            completion_times[comp_name] = task["End"]

    # Calculate tardiness and priority-weighted completion
    for comp in components:
        comp_name = comp["name"]
        if comp_name in completion_times:
            completion = completion_times[comp_name]
            due_date = comp["due_date"]

            # Calculate tardiness (if any)
            if completion.date() > due_date.date():
                days_late = (completion.date() - due_date.date()).days
                tardiness += days_late

            # Priority-weighted completion time (higher priority = higher weight)
            priority_weight = comp["priority"] / 10  # Normalize to 0-1
            completion_time_hours = (completion - SHIFT_START).total_seconds() / 3600
            priority_weighted_completion += completion_time_hours * priority_weight

    # Calculate makespan (total production time)
    if schedule:
        start_times = [task["Start"] for task in schedule]
        end_times = [task["End"] for task in schedule]
        earliest_start = min(start_times)
        latest_end = max(end_times)
        makespan = (latest_end - earliest_start).total_seconds() / 3600  # in hours

    # Calculate machine idle time
    machine_busy_times = {machine: 0 for machine in machines}
    for task in schedule:
        machine = task["Machine"]
        duration = (task["End"] - task["Start"]).total_seconds() / 60  # in minutes
        machine_busy_times[machine] += duration

    # Calculate total available machine time
    if schedule:
        start_date = min(task["Start"] for task in schedule).date()
        end_date = max(task["End"] for task in schedule).date()
        num_days = (end_date - start_date).days + 1

        shift_duration = (SHIFT_END - SHIFT_START).total_seconds() / 60  # in minutes
        total_available_time = shift_duration * num_days * len(machines)

        # Calculate total idle time
        total_busy_time = sum(machine_busy_times.values())
        machine_idle_time = total_available_time - total_busy_time

    # Combined fitness score (lower is better)
    # Weighting factors can be adjusted based on priorities
    tardiness_weight = 50
    makespan_weight = 10
    idle_time_weight = 5
    priority_weight = 20

    fitness_score = (
        tardiness_weight * tardiness +
        makespan_weight * makespan +
        idle_time_weight * (machine_idle_time / 1000) +  # Scale down idle time
        priority_weight * priority_weighted_completion
    )

    # Higher fitness is better for PyGAD, so we invert the score
    return 10000 / (fitness_score + 1)  # Add 1 to avoid division by zero

def fitness_func(ga_instance, solution, solution_idx):
    """Fitness function for the genetic algorithm."""
    global all_components

    # Decode the solution chromosome into component order only
    # No longer reordering operations within components to preserve operation sequence
    component_order = decode_chromosome(solution)

    # Create a modified list of components based on the genetic algorithm solution
    # Only reordering components, not operations within components
    modified_components = reorder_components(all_components, component_order)

    # Schedule using the modified components
    schedule = schedule_jobs_internal(modified_components)

    # Evaluate the schedule quality
    fitness = evaluate_schedule(schedule, modified_components)

    return fitness

def decode_chromosome(chromosome):
    """Decode a chromosome into component order only, preserving operation sequence."""
    global all_components

    # Component order genes
    n_components = len(all_components)
    component_genes = chromosome[:n_components]

    # Create order based on the value of the genes
    component_order = np.argsort(component_genes)

    return component_order

def reorder_components(original_components, component_order):
    """Create a new component list with the ordering from GA, but preserving operation sequence."""
    # Create a deep copy of components to avoid modifying the original
    reordered_components = []

    # Reorder components based on component_order
    for idx in component_order:
        comp = original_components[idx].copy()  # Shallow copy of the component dict
        # No operation reordering - maintain the original operation sequence
        reordered_components.append(comp)

    return reordered_components

def schedule_jobs_internal(components):
    """Internal scheduling function used by both direct scheduling and GA."""
    # Assign darker colors to each component
    for comp in components:
        if comp["name"] not in COMPONENT_COLORS:
            # Generate darker shades for components
            r = random.randint(20, 120)
            g = random.randint(20, 120)
            b = random.randint(20, 120)
            COMPONENT_COLORS[comp["name"]] = f"#{r:02x}{g:02x}{b:02x}"

    schedule = []
    machine_available_time = {machine: SHIFT_START for machine in machines}
    last_component_on_machine = {machine: None for machine in machines}

    start_date = datetime.now().replace(hour=SHIFT_START.hour, minute=SHIFT_START.minute, second=0, microsecond=0)

    # Create a dictionary to track the next operation index for each component instance
    component_operation_progress = {}

    # Process each component
    for component in components:
        component_name = component["name"]
        operations = component["operations"]
        cycle_times = component["cycle_times"]
        setup_times = component["setup_times"]
        quantity = component["quantity"]

        # Process each unit of the component
        for i in range(quantity):
            job_id = f"{component_name}-{i + 1}"
            component_operation_progress[job_id] = 0  # Initialize operation progress for this job
            job_ready_time = start_date

            # Process operations in sequence as specified in the component definition
            while component_operation_progress[job_id] < len(operations):
                # Get the next operation for this job based on the sequence
                op_idx = component_operation_progress[job_id]
                machine_name = operations[op_idx]

                op_time = cycle_times[machine_name]
                setup_time = 0

                # If this is a different component from the last one on this machine, add setup time
                if last_component_on_machine[machine_name] != component_name:
                    setup_time = setup_times[machine_name]
                    last_component_on_machine[machine_name] = component_name

                # Determine when both the machine and the job are ready
                earliest_start = max(machine_available_time[machine_name], job_ready_time)

                # Make sure the start time is within a shift
                if not is_within_shift(earliest_start):
                    earliest_start = next_working_time(earliest_start)

                total_time = setup_time + op_time
                end_time = earliest_start + timedelta(minutes=total_time)

                # If operation goes beyond shift, adjust to next shift
                if not is_within_shift(end_time):
                    remaining_time = total_time
                    curr_time = earliest_start

                    # Process as much as we can in this shift
                    curr_shift_end = datetime.combine(curr_time.date(), SHIFT_END.time())
                    time_available = (curr_shift_end - curr_time).total_seconds() / 60

                    if time_available > 0:
                        # Process partial job in current shift
                        partial_end = curr_time + timedelta(minutes=min(remaining_time, time_available))

                        # If we already had setup time, it's done in the current shift
                        if setup_time > 0:
                            partial_time = min(remaining_time, time_available)
                            setup_used = min(setup_time, partial_time)
                            op_used = partial_time - setup_used

                            schedule.append({
                                "Component": component_name,
                                "Machine": machine_name,
                                "Operation": machines[machine_name]["operation"],
                                "OperationSequence": op_idx + 1,  # Add operation sequence number
                                "Start": curr_time,
                                "End": partial_end,
                                "Setup Time": setup_used,
                                "Cycle Time": op_used,
                                "Job": job_id,
                                "Status": "Partial" if partial_end < end_time else "Complete"
                            })

                            # Update remaining times
                            setup_time -= setup_used
                            op_time -= op_used
                        else:
                            # Just cycle time used
                            op_used = min(op_time, time_available)

                            schedule.append({
                                "Component": component_name,
                                "Machine": machine_name,
                                "Operation": machines[machine_name]["operation"],
                                "OperationSequence": op_idx + 1,  # Add operation sequence number
                                "Start": curr_time,
                                "End": partial_end,
                                "Setup Time": 0,
                                "Cycle Time": op_used,
                                "Job": job_id,
                                "Status": "Partial" if partial_end < end_time else "Complete"
                            })

                            # Update remaining time
                            op_time -= op_used

                        remaining_time -= time_available
                        curr_time = next_working_time(partial_end)
                    else:
                        # Move to next shift
                        curr_time = next_working_time(curr_time)

                    # Continue job in next shift if needed
                    if remaining_time > 0:
                        end_time = curr_time + timedelta(minutes=remaining_time)

                        schedule.append({
                            "Component": component_name,
                            "Machine": machine_name,
                            "Operation": machines[machine_name]["operation"],
                            "OperationSequence": op_idx + 1,  # Add operation sequence number
                            "Start": curr_time,
                            "End": end_time,
                            "Setup Time": setup_time,  # Remaining setup time, if any
                            "Cycle Time": op_time,     # Remaining op time
                            "Job": job_id,
                            "Status": "Complete"
                        })
                else:
                    # Operation fits within shift
                    schedule.append({
                        "Component": component_name,
                        "Machine": machine_name,
                        "Operation": machines[machine_name]["operation"],
                        "OperationSequence": op_idx + 1,  # Add operation sequence number
                        "Start": earliest_start,
                        "End": end_time,
                        "Setup Time": setup_time,
                        "Cycle Time": op_time,
                        "Job": job_id,
                        "Status": "Complete"
                    })

                # Update availability for next operation
                machine_available_time[machine_name] = end_time
                job_ready_time = end_time

                # Move to the next operation in sequence for this job
                component_operation_progress[job_id] += 1

    # Sort the schedule by start time for better readability
    schedule.sort(key=lambda x: x["Start"])
    return schedule

def schedule_jobs(components):
    """Schedule multiple components using genetic algorithm."""
    global all_components, ga_instance, best_solution

    # Store components globally for use in fitness function
    all_components = components.copy()

    # Calculate chromosome length - only component order genes now
    # We're not reordering operations within components
    n_components = len(components)
    chromosome_length = n_components

    print(f"\nInitializing Genetic Algorithm with chromosome length: {chromosome_length}")
    print("Please wait, this might take some time...")

    # Create an instance of the GA
    ga_instance = pygad.GA(
        num_generations=50,
        num_parents_mating=10,
        fitness_func=fitness_func,
        sol_per_pop=20,
        num_genes=chromosome_length,
        gene_type=float,
        init_range_low=0.0,
        init_range_high=1.0,
        parent_selection_type="tournament",
        K_tournament=5,
        crossover_type="two_points",
        mutation_type="random",
        mutation_percent_genes=10,
        keep_elitism=2,
        stop_criteria=["reach_50"]
    )

    # Run the GA
    ga_instance.run()

    # After the GA finishes, get the best solution
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    best_solution = solution

    print(f"\nGA optimization complete. Best fitness: {solution_fitness}")

    # Decode the solution chromosome into component order only
    component_order = decode_chromosome(best_solution)

    # Create a modified list of components based on the genetic algorithm solution
    modified_components = reorder_components(components, component_order)

    # Schedule using the modified components
    return schedule_jobs_internal(modified_components)

def display_schedule(schedule):
    """Display the production schedule."""
    print("\n=== Production Schedule ===")
    # Sort by component, job and operation sequence for more readable output
    sorted_schedule = sorted(schedule, key=lambda x: (x['Component'], x['Job'], x['OperationSequence']))

    for task in sorted_schedule:
        status = f"({task['Status']})" if 'Status' in task else ""
        print(f"{task['Component']} | Operation {task['OperationSequence']} | {task['Machine']} ({task['Operation']}) | "
              f"Start: {task['Start'].strftime('%Y-%m-%d %H:%M')} | End: {task['End'].strftime('%Y-%m-%d %H:%M')} | "
              f"Setup: {task['Setup Time']} mins | Cycle: {task['Cycle Time']} mins {status}")

def plot_gantt_chart(schedule):
    """Generate a Gantt chart for the schedule."""
    print("\nGenerating Gantt chart...")
    fig, ax = plt.subplots(figsize=(15, 8))

    y_labels = list(machines.keys())
    y_pos = {machine: i for i, machine in enumerate(y_labels)}

    # Get earliest and latest dates for chart bounds
    start_dates = [task['Start'] for task in schedule]
    end_dates = [task['End'] for task in schedule]
    chart_start = min(start_dates) - timedelta(minutes=30)
    chart_end = max(end_dates) + timedelta(minutes=30)

    # Plot shift boundaries
    unique_dates = sorted(list(set([d.date() for d in start_dates + end_dates])))
    for date in unique_dates:
        shift_start = datetime.combine(date, SHIFT_START.time())
        shift_end = datetime.combine(date, SHIFT_END.time())

        # Plot shift starts and ends
        ax.axvline(x=mdates.date2num(shift_start), color='darkgray', linestyle='--', alpha=0.7)
        ax.axvline(x=mdates.date2num(shift_end), color='darkgray', linestyle='--', alpha=0.7)

    # Group by component and machine
    for task in schedule:
        start = mdates.date2num(task['Start'])
        end = mdates.date2num(task['End'])

        # Use component color for fill and machine color for edge
        component_color = COMPONENT_COLORS.get(task["Component"], "gray")
        machine_color = MACHINE_COLORS.get(task["Machine"], "black")

        # Create a bar with component color fill and machine color edge
        ax.barh(
            y_pos[task["Machine"]],
            width=end - start,
            left=start,
            height=0.6,
            color=component_color,     # Component color for fill
            edgecolor='black',   # Machine color for border
            linewidth=0.5,
            alpha=0.8
        )

        # Add component name and operation sequence text label
        label_text = f"{task['Component']} (Op {task['OperationSequence']})"
        ax.text(
            start + (end - start) / 2,
            y_pos[task["Machine"]],
            label_text,
            ha='center',
            va='center',
            fontsize=5,
            color='white',  # White text for better visibility on darker backgrounds
            fontweight='bold'
        )

        # Add start time and end time labels
        start_time_str = task['Start'].strftime('%H:%M')
        end_time_str = task['End'].strftime('%H:%M')

        # Add start time label at the beginning of the bar
        ax.text(
            start,
            y_pos[task["Machine"]] - 0.3,
            start_time_str,
            ha='center',
            va='top',
            fontsize=7,
            color='black',
            rotation=90
        )

        # Add end time label at the end of the bar
        ax.text(
            end,
            y_pos[task["Machine"]] - 0.3,
            end_time_str,
            ha='center',
            va='top',
            fontsize=7,
            color='black',
            rotation=90
        )

    # Create a legend for components
    from matplotlib.patches import Patch
    legend_elements = [
      Patch(facecolor=color, edgecolor='black', label=comp)
      for comp, color in COMPONENT_COLORS.items()
    ]
    ax.legend(handles=legend_elements, loc='upper right', title='Components')

    # Set chart properties
    ax.set_yticks(list(y_pos.values()))
    ax.set_yticklabels(y_labels)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%d-%b'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))

    plt.xlabel("Time")
    plt.title("Multi-Component Gantt Chart with Operation Sequences")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Set x-axis limits to focus on the relevant timeframe
    ax.set_xlim(mdates.date2num(chart_start), mdates.date2num(chart_end))

    plt.show()

def export_schedule(schedule, filename="production_schedule.csv"):
    """Export the schedule to a CSV file."""
    df = pd.DataFrame(schedule)

    # Convert datetime objects to string for CSV export
    df['Start'] = df['Start'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M'))
    df['End'] = df['End'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M'))

    df.to_csv(filename, index=False)
    print(f"\nSchedule exported to {filename}")

def analyze_schedule(schedule, components):
    """Analyze the schedule and provide insights."""
    # Organize components information
    comp_info = {}
    for comp in components:
        comp_info[comp["name"]] = {
            "due_date": comp["due_date"],
            "priority": comp["priority"],
            "quantity": comp["quantity"]
        }

    # Calculate completion times for each component
    completion_times = {}
    for task in schedule:
        comp_name = task["Component"]
        if comp_name not in completion_times or task["End"] > completion_times[comp_name]:
            completion_times[comp_name] = task["End"]

    # Calculate statistics for each component
    analysis = []
    for comp_name, complete_time in completion_times.items():
        due_date = comp_info[comp_name]["due_date"]
        is_on_time = complete_time.date() <= due_date.date()
        days_diff = (due_date.date() - complete_time.date()).days

        status = "On Time" if is_on_time else f"Late by {abs(days_diff)} days"

        analysis.append({
            "Component": comp_name,
            "Completion Time": complete_time,
            "Due Date": due_date,
            "Status": status,
            "Priority": comp_info[comp_name]["priority"],
            "Quantity": comp_info[comp_name]["quantity"]
        })

    # Calculate machine utilization
    machine_usage = {machine: timedelta(0) for machine in machines}
    for task in schedule:
        machine = task["Machine"]
        duration = task["End"] - task["Start"]
        machine_usage[machine] += duration

    # Calculate total shift time available
    start_date = min(task["Start"] for task in schedule).date()
    end_date = max(task["End"] for task in schedule).date()
    num_days = (end_date - start_date).days + 1

    shift_duration = SHIFT_END - SHIFT_START
    total_available_time = shift_duration * num_days

    # Calculate utilization percentages
    utilization = {}
    for machine, usage in machine_usage.items():
        utilization[machine] = (usage.total_seconds() / total_available_time.total_seconds()) * 100

    # Display analysis
    print("\n=== Schedule Analysis ===")

    print("\nComponent Completion Status:")
    for item in analysis:
        print(f"{item['Component']} (Priority {item['Priority']}, Qty {item['Quantity']}): "
              f"Completes on {item['Completion Time'].strftime('%Y-%m-%d %H:%M')} | "
              f"Due: {item['Due Date'].strftime('%Y-%m-%d')} | Status: {item['Status']}")

    print("\nMachine Utilization:")
    for machine, percent in utilization.items():
        print(f"{machine}: {percent:.2f}%")

    # Compute earliest possible completion time
    earliest_completion = min(task["Start"] for task in schedule)
    latest_completion = max(task["End"] for task in schedule)
    total_duration = latest_completion - earliest_completion

    print(f"\nProduction starts: {earliest_completion.strftime('%Y-%m-%d %H:%M')}")
    print(f"Production ends: {latest_completion.strftime('%Y-%m-%d %H:%M')}")
    print(f"Total production time: {total_duration}")

def main():
    """Main function to run the shop floor scheduler."""
    print("\n*** Shop Floor Scheduling System ***\n")

    # Get user inputs
    components = get_user_inputs()

    # Use genetic algorithm approach
    print("\nRunning Genetic Algorithm optimization...")
    ga_schedule = schedule_jobs(components)

    # Display and analyze results
    display_schedule(ga_schedule)
    analyze_schedule(ga_schedule, components)
    plot_gantt_chart(ga_schedule)

    # Ask for schedule export
    print("\nWould you like to export the schedule to CSV? (y/n)")
    export = input("> ")
    if export.lower() == 'y':
        export_schedule(ga_schedule)

# Run the main function if this script is executed directly
if __name__ == "__main__":
    main()