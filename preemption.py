import re
from datetime import datetime
import sys

filename = sys.argv[1]

with open(filename) as f:
  log = f.read()

pattern = r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}'
times = re.findall(pattern, log)

datetimes = [datetime.strptime(t, '%Y-%m-%dT%H:%M:%S') for t in times]
elapsed = []
for i in range(1, len(datetimes)):
  elapsed.append((datetimes[i] - datetimes[i-1]).seconds / 60)
  # print(f"{elapsed} minutes between preemptions")
print(elapsed)
# plot the histogram
import matplotlib.pyplot as plt
num_bins = 15
bins_upper_bound = int(max(elapsed)) + 20
plt.hist(elapsed, bins=range(0, bins_upper_bound, bins_upper_bound // num_bins))
plt.xlabel("Minutes between preemptions")
plt.ylabel("Number of preemptions")
plt.title(f"Preemption frequency for {filename}")
plt.show()


# 3-8
# scp -i ~/.ssh/stability_ai_hpc itsleonwu@welcomewest.hpc.stability.ai:/fsx/itsleonwu/audiolm-pytorch-results/error-7983.log .
# scp -i ~/.ssh/stability_ai_hpc itsleonwu@welcomewest.hpc.stability.ai:/fsx/itsleonwu/audiolm-pytorch-results/error-7984.log .
# scp -i ~/.ssh/stability_ai_hpc itsleonwu@welcomewest.hpc.stability.ai:/fsx/itsleonwu/audiolm-pytorch-results/error-7985.log .
# scp -i ~/.ssh/stability_ai_hpc itsleonwu@welcomewest.hpc.stability.ai:/fsx/itsleonwu/audiolm-pytorch-results/error-7986.log .
# scp -i ~/.ssh/stability_ai_hpc itsleonwu@welcomewest.hpc.stability.ai:/fsx/itsleonwu/audiolm-pytorch-results/error-7987.log .
# scp -i ~/.ssh/stability_ai_hpc itsleonwu@welcomewest.hpc.stability.ai:/fsx/itsleonwu/audiolm-pytorch-results/error-7988.log .
