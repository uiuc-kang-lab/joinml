import logging
import json


class Estimates:
    def __init__(self, cost: float, gt: float, est: float, lbs, ubs) -> None:
        self.gt = gt
        self.est = est
        self.lbs = lbs
        self.ubs = ubs
        self.true_error = (self.est - self.gt) / self.gt
        self.lb_errors = [(lb - self.gt) / self.gt for lb in self.lbs]
        self.ub_errors = [(ub - self.gt) / self.gt for ub in self.ubs]
        self.cost = cost
        self.coverages = [int(self.gt >= lb and self.gt <= ub) for lb, ub in zip(self.lbs, self.ubs)]

    def log(self):
        logging.info("budget: {} true_error {} lb_errors {} ub_errors {} coverages {}".format(
            self.cost, self.true_error, self.lb_errors, self.ub_errors, self.coverages))
    
    def save(self, output_file: str, surfix: str = ""):
        output_file = output_file.split(".")[0] + surfix + ".jsonl"
        with open(f"{output_file}", "a+") as f:
            json.dump({
                "budget": float(self.cost),
                "gt": float(self.gt),
                "est": float(self.est),
                "true_error": float(self.true_error),
                "lbs": self.lbs,
                "ubs": self.ubs,
                "lb_errors": self.lb_errors,
                "ub_error": self.ub_errors,
                "coverages": self.coverages
                }, f)
            f.write("\n")
