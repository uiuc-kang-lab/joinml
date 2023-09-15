import matplotlib.pyplot as plt
import csv

errors = []
costs = []
sample_ratios = []
with open("ripple_join.csv") as f:
    reader = csv.reader(f)
    _ = next(reader)
    for row in reader:
        sample_ratio, _, _, error, oracle_calls = row
        sample_ratios.append(float(sample_ratio))
        errors.append(float(error))
        costs.append(1-float(oracle_calls))

plt.plot(sample_ratios, errors)
plt.plot(sample_ratios, costs)
plt.legend(["count(*) error", "% of oracle calls saved"])
plt.savefig("ripple.png", dpi=300)