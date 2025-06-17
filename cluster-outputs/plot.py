import re
import matplotlib.pyplot as plt

log_file_path = "/home/ulzhalgas/sgcrl/cluster-outputs/txt/sgcrl-bin/bin11881997-15-mil.txt"
output_image_path = "/home/ulzhalgas/sgcrl/cluster-outputs/plots/sgcrl-bin/success_vs_steps-new2.png"

# Lists to store data
actor_steps = []
success_rates = []

# Regex to extract steps and success
pattern = re.compile(r"Actor Steps = (\d+).*?Success 1000 = ([\d\.]+)")

# Read and parse the file
with open(log_file_path, "r") as file:
    for line in file:
        match = pattern.search(line)
        if match:
            steps = int(match.group(1))
            success = float(match.group(2))
            actor_steps.append(steps)
            success_rates.append(success)


# Plotting
plt.figure(figsize=(10, 6))
plt.plot(actor_steps, success_rates, marker="o", linestyle="-", color="blue")
plt.xlabel("Actor Steps")
plt.ylabel("Success")
plt.title("Success Rate vs Actor Steps")
plt.yticks([0.0, 0.1, 0.2, 0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]) 
plt.grid(True)
plt.tight_layout()

# Save plot
plt.savefig(output_image_path)
print(f"Plot saved to: {output_image_path}")