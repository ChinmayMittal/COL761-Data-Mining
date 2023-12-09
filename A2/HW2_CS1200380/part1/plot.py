import re
import matplotlib.pyplot as plt

# Define regular expressions to match support values and time values
support_pattern = re.compile(r"Support (\d+)")
time_pattern = re.compile(r"real\t(\d+m\d+\.\d+s)")

# Initialize lists to store support values and algorithm times
support_values = []
fsg_times = []
gspan_times = []
gaston_times = []

# Read the results file
with open("time_output.txt", "r") as file:
    lines = file.readlines()

# Parse the results
current_support = None
current_algorithm = None
algorithms = ("fsg", "gspan", "gaston")
for line in lines:
    support_match = support_pattern.match(line)
    time_match = time_pattern.match(line)
    for algo in algorithms:
        if algo in line:
            current_algorithm = algo

    if support_match:
        current_support = int(support_match.group(1))
        support_values.append(current_support)
    elif time_match:
        time = time_match.group(1)
        if current_algorithm == "fsg" :
            fsg_times.append(float(time.split("m")[0]) * 60 + float(time.split("m")[1][:-1]))
        elif current_algorithm == "gspan":
            gspan_times.append(float(time.split("m")[0]) * 60 + float(time.split("m")[1][:-1]))
        elif current_algorithm == "gaston":
            gaston_times.append(float(time.split("m")[0]) * 60 + float(time.split("m")[1][:-1]))

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(support_values, fsg_times, marker='o', label='fsg')
plt.plot(support_values, gspan_times, marker='o', label='gspan')
plt.plot(support_values, gaston_times, marker='o', label='gaston')
plt.xlabel('Support Values')
plt.ylabel('Running Time (seconds)')
plt.title('Running Time of Algorithms for Different Support Values')
plt.legend()
plt.grid(True)

# Save the plot as an image or display it
plt.savefig('running_time_plot.png')
# plt.show()  # Uncomment to display the plot interactively

# Close the plot
plt.close()
