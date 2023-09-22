import matplotlib.pyplot as plt
import csv

# errors = []
# costs = []
# sample_ratios = []
# with open("random_naive.csv") as f:
#     reader = csv.reader(f)
#     _ = next(reader)
#     for row in reader:
#         sample_ratio, _, _, error, oracle_calls = row
#         sample_ratios.append(float(sample_ratio))
#         errors.append(float(error))
#         costs.append(1-float(oracle_calls))

# plt.plot(sample_ratios, errors)
# plt.plot(sample_ratios, costs)
# plt.legend(["count(*) error", "% of oracle calls saved"])
# plt.savefig("ripple.png", dpi=300)


def plot_random_naive():
    groundtruth = 1144
    total_pairs = 3500 * 3500
    # sample_ratios = []
    # count_results = []
    # ci_uppers = []
    # ci_lowers = []
    data = []
    with open("random_naive.csv") as f:
        reader = csv.reader(f)
        _ = next(reader)
        for row in reader:
            sample_ratio, mean, ci_lower, ci_upper = row
            sample_ratio = float(sample_ratio)
            count_result = float(mean) * total_pairs
            ci_lower = float(ci_lower) * total_pairs
            ci_upper = float(ci_upper) * total_pairs
            data.append([sample_ratio, count_result, ci_lower, ci_upper])
    data.sort(key=lambda l: l[0])
    sample_ratios, count_results, ci_lowers, ci_uppers = list(zip(*data))
    
    plt.xscale("log")
    plt.plot(sample_ratios, count_results)
    plt.fill_between(sample_ratios, ci_lowers, ci_uppers, color='b', alpha=.1)
    plt.axhline(groundtruth, color='r', linestyle='-')
    plt.legend(["AQP result", "95% confidence interval", "groundtruth"])
    plt.xlabel("sampling ratio")
    plt.ylabel("count results")
    plt.savefig("random_naive.png", dpi=300)
    plt.show()

def plot_ripple():
    groundtruth = 1144
    total_pairs = 3500 * 3500
    # sample_ratios = []
    # count_results = []
    # ci_uppers = []
    # ci_lowers = []
    data = []
    with open("ripple.csv") as f:
        reader = csv.reader(f)
        _ = next(reader)
        for row in reader:
            sample_ratio, mean, ci_lower, ci_upper = row
            sample_ratio = float(sample_ratio)
            count_result = float(mean) * total_pairs
            ci_lower = float(ci_lower) * total_pairs
            ci_upper = float(ci_upper) * total_pairs
            data.append([sample_ratio, count_result, ci_lower, ci_upper])
    data.sort(key=lambda l: l[0])
    sample_ratios, count_results, ci_lowers, ci_uppers = list(zip(*data))
    
    plt.xscale("log")
    plt.plot(sample_ratios, count_results)
    plt.fill_between(sample_ratios, ci_lowers, ci_uppers, color='b', alpha=.1)
    plt.axhline(groundtruth, color='r', linestyle='-')
    plt.legend(["AQP result", "95% confidence interval", "groundtruth"])
    plt.xlabel("sampling ratio")
    plt.ylabel("count results")
    plt.savefig("ripple.png", dpi=300)
    plt.show() 

if __name__ == '__main__':
    plot_ripple()