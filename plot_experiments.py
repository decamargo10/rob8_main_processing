import matplotlib.pyplot as plt

data = '''Yellow_tool: Detected=8, Not Detected=0, Accuracy=100.0%
Multimeter: Detected=8, Not Detected=3, Accuracy=72.72727272727273%
Tape: Detected=9, Not Detected=3, Accuracy=75.0%
Ninja: Detected=0, Not Detected=10, Accuracy=0.0%
Screwdriver: Detected=4, Not Detected=6, Accuracy=40.0%
Hot_glue: Detected=6, Not Detected=8, Accuracy=42.857142857142854%
Overall: Total Detected=35, Total Not Detected=30, Overall Accuracy=53.84615384615385%'''

lines = data.split('\n')
items = []
accuracies = []
colors = []

# Extract overall accuracy separately
overall_line = lines[-1]
overall_accuracy = float(overall_line.split('=')[-1].strip('%'))
items.append('Overall')
accuracies.append(overall_accuracy)
colors.append('red')

for line in lines[:-1]:  # exclude the overall accuracy
    if ':' not in line:
        continue
    parts = line.split(':')
    item = parts[0].strip()
    accuracy = float(parts[-1].split('=')[-1].strip('%'))
    if item != 'Overall':
        items.append(item)
        accuracies.append(accuracy)
        colors.append('blue')


# Set figure size
fig, ax = plt.subplots(figsize=(8.15, 4.45))

plt.bar(items, accuracies, color=colors)
plt.ylabel('Accuracy in %')
plt.title('Intention recognition accuracy on cluttered scene using GRU(4)')
plt.ylim(0, 100)


# Save figure with specified dimensions
plt.savefig('accuracy_plot_gru_5.png', dpi=100, bbox_inches='tight')
plt.show()
