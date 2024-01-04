import matplotlib.pyplot as plt

# Define the path to the data file
data_file_path = 'training_data.txt'

# Initialize lists to store the extracted data
averages = []
max_points = []
min_points = []

# Read the data from the file
with open(data_file_path, 'r') as file:
    for line in file:
        # Split the line into parts based on space
        parts = line.split()

        # Extract the average, max, and min values from the line

        average = float(parts[6])
        max_point = int(parts[8])
        min_point = int(parts[10])

        # Append the extracted values to the respective lists
        averages.append(average)
        max_points.append(max_point)
        min_points.append(min_point)

# Plotting the data
plt.figure(figsize=(10, 6))
plt.plot(averages, label='Average Points')
plt.plot(max_points, label='Max Points')
plt.plot(min_points, label='Min Points')

# Set the font to Times New Roman
plt.rc('font', family='Times New Roman')
# Add titles and labels
# plt.title('Data of Training Process', fontsize=17, fontname='Times New Roman')
plt.xlabel('Game Intervals (100)', fontsize=14, fontname='Times New Roman')
plt.ylabel('Points', fontsize=14, fontname='Times New Roman')

# Show legend
plt.legend()

# Show the plot
plt.show()
