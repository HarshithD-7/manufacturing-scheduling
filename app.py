from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import random
import pandas as pd

# Constants for shift
SHIFT_START = datetime.strptime("09:00", "%H:%M").replace(year=2025, month=5, day=14)
SHIFT_END = datetime.strptime("17:30", "%H:%M").replace(year=2025, month=5, day=14)

machines = {
    "VMC 1": {"operation": "Milling", "available_from": SHIFT_START},
    "VMC 2": {"operation": "Milling", "available_from": SHIFT_START},
    "Turning Center 1": {"operation": "Turning", "available_from": SHIFT_START},
    "Turning Center 2": {"operation": "Turning", "available_from": SHIFT_START},
    "Surface Grinder": {"operation": "Surface Grinding", "available_from": SHIFT_START},
    "Cutter": {"operation": "Cutter Operation", "available_from": SHIFT_START},
    "ID Grinder": {"operation": "ID Grinding", "available_from": SHIFT_START},
}

# Static colors for machine types to ensure consistency in the Gantt chart
MACHINE_COLORS = {
    "VMC": "skyblue",
    "Turning Center": "lightgreen",
    "Surface Grinder": "salmon",
    "Cutter": "orange",
    "ID Grinder": "plum"
}

# Component colors for differentiation in the Gantt chart
COMPONENT_COLORS = {}

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
    print("=== Multi-Component Shopfloor Scheduler ===")
    
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
            "operations": operations,
            "cycle_times": cycle_times,
            "setup_times": setup_times,
        })
    
    print("\nSort components by:")
    print("1. Due Date (earliest first)")
    print("2. Priority (highest first)")
    print("3. Shortest Processing Time")
    sort_option = input("Enter your choice (1-3): ")
    
    scheduling_strategy = None
    if sort_option == '1':
        scheduling_strategy = "due_date"
    elif sort_option == '2':
        scheduling_strategy = "priority"
    elif sort_option == '3':
        scheduling_strategy = "processing_time"
    else:
        print("Invalid choice. Defaulting to Due Date.")
        scheduling_strategy = "due_date"
    
    return components, scheduling_strategy

def sort_components(components, strategy):
    """Sort components based on chosen strategy."""
    if strategy == "due_date":
        return sorted(components, key=lambda x: x["due_date"])
    elif strategy == "priority":
        return sorted(components, key=lambda x: x["priority"], reverse=True)
    elif strategy == "processing_time":
        # Calculate total processing time for each component
        for comp in components:
            total_time = sum(comp["cycle_times"][op] + comp["setup_times"][op] for op in comp["operations"])
            comp["total_processing_time"] = total_time
        return sorted(components, key=lambda x: x["total_processing_time"])
    return components

def schedule_jobs(components, strategy):
    """Schedule multiple components based on strategy."""
    components = sort_components(components, strategy)
    
    # Assign random colors to each component
    for comp in components:
        if comp["name"] not in COMPONENT_COLORS:
            r = random.randint(100, 200)
            g = random.randint(100, 200)
            b = random.randint(100, 200)
            COMPONENT_COLORS[comp["name"]] = f"#{r:02x}{g:02x}{b:02x}"
    
    schedule = []
    machine_available_time = {machine: SHIFT_START for machine in machines}
    last_component_on_machine = {machine: None for machine in machines}
    
    start_date = datetime.now().replace(hour=SHIFT_START.hour, minute=SHIFT_START.minute, second=0, microsecond=0)
    
    # Process each component
    for component in components:
        component_name = component["name"]
        operations = component["operations"]
        cycle_times = component["cycle_times"]
        setup_times = component["setup_times"]
        quantity = component["quantity"]
        
        # Process each unit of the component
        for i in range(quantity):
            job_ready_time = start_date
            
            # Process each operation for this component unit
            for machine_name in operations:
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
                                "Start": curr_time,
                                "End": partial_end,
                                "Setup Time": setup_used,
                                "Cycle Time": op_used,
                                "Job": f"{component_name}-{i + 1}",
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
                                "Start": curr_time,
                                "End": partial_end,
                                "Setup Time": 0,
                                "Cycle Time": op_used,
                                "Job": f"{component_name}-{i + 1}",
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
                            "Start": curr_time,
                            "End": end_time,
                            "Setup Time": setup_time,  # Remaining setup time, if any
                            "Cycle Time": op_time,     # Remaining op time
                            "Job": f"{component_name}-{i + 1}",
                            "Status": "Complete"
                        })
                else:
                    # Operation fits within shift
                    schedule.append({
                        "Component": component_name,
                        "Machine": machine_name,
                        "Operation": machines[machine_name]["operation"],
                        "Start": earliest_start,
                        "End": end_time,
                        "Setup Time": setup_time,
                        "Cycle Time": op_time,
                        "Job": f"{component_name}-{i + 1}",
                        "Status": "Complete"
                    })
                
                # Update availability for next operation
                machine_available_time[machine_name] = end_time
                job_ready_time = end_time
    
    return schedule

def display_schedule(schedule):
    """Display the production schedule."""
    print("\n=== Production Schedule ===")
    for task in schedule:
        status = f"({task['Status']})" if 'Status' in task else ""
        print(f"{task['Component']} | {task['Machine']} ({task['Operation']}) | "
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
        
        # Use machine color for the bar
        machine_color = MACHINE_COLORS.get(task["Machine"], "gray")
        
        # Add component color as edge
        component_color = COMPONENT_COLORS.get(task["Component"], "black")
        
        # Create a bar with machine color and component edgecolor
        ax.barh(
            y_pos[task["Machine"]],
            width=end - start,
            left=start,
            height=0.6,
            color=machine_color,
            edgecolor=component_color,
            linewidth=2,
            alpha=0.8
        )
        
        # Add component name text label in the center
        ax.text(
            start + (end - start) / 2,
            y_pos[task["Machine"]],
            task["Component"],
            ha='center',
            va='center',
            fontsize=8,
            color='black',
            fontweight='bold'
        )
        
        # Add start time and end time labels (NEW)
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
        Patch(facecolor='white', edgecolor=color, label=comp)
        for comp, color in COMPONENT_COLORS.items()
    ]
    ax.legend(handles=legend_elements, loc='upper right', title='Components')
    
    # Set chart properties
    ax.set_yticks(list(y_pos.values()))
    ax.set_yticklabels(y_labels)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%d-%b'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    
    plt.xlabel("Time")
    plt.title("Multi-Component Shopfloor Gantt Chart")
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
    """Main function to run the scheduler."""
    components, strategy = get_user_inputs()
    schedule = schedule_jobs(components, strategy)
    display_schedule(schedule)
    analyze_schedule(schedule, components)
    
    export_choice = input("\nExport schedule to CSV? (y/n): ")
    if export_choice.lower() == 'y':
        export_schedule(schedule)
    
    show_gantt = input("Show Gantt chart? (y/n): ")
    if show_gantt.lower() == 'y':
        plot_gantt_chart(schedule)

if __name__ == "__main__":
    main()