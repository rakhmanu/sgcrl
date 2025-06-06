import re
import matplotlib.pyplot as plt

log_file_path = "/home/ulzhalgas/sgcrl/cluster-outputs/txt/crl-drawer/drawer11798527.txt"
output_image_path = "/home/ulzhalgas/sgcrl/cluster-outputs/plots/crl-drawer/success_vs_steps.png"

# Lists to store data
actor_steps = []
success_rates = []

# Regex to extract steps and success
pattern = re.compile(r"Actor Steps = (\d+).*?Success = ([\d\.]+)")

# Read and parse the file
with open(log_file_path, "r") as file:
    for line in file:
        match = pattern.search(line)
        if match:
            steps = int(match.group(1))
            success = float(match.group(2))
            actor_steps.append(steps)
            success_rates.append(success)

# Normalize actor steps to start from zero
if actor_steps:
    min_step = min(actor_steps)
    normalized_steps = [s - min_step for s in actor_steps]
else:
    print("No data matched the pattern.")
    exit()

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(normalized_steps, success_rates, marker="o", linestyle="-", color="blue")
plt.xlabel("Actor Steps")
plt.ylabel("Success")
plt.title("Success Rate vs Actor Steps")
plt.grid(True)
plt.tight_layout()

# Save plot
plt.savefig(output_image_path)
print(f"Plot saved to: {output_image_path}")