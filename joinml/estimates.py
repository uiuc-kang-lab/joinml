import logging
import json


class Estimates:
    def __init__(self, budget: int, gt: float, est: float, lb: float, ub: float) -> None:
        self.gt = gt
        self.est = est
        self.lb = lb
        self.ub = ub
        self.true_error = (self.est - self.gt) / self.gt
        self.lb_error = (self.lb - self.gt) / self.gt
        self.ub_error = (self.ub - self.gt) / self.gt
        self.budget = budget

    def log(self):
        logging.info("budget: {} gt: {} est: {} lb: {} ub: {} true_error {} lb_error {} ub_error {}".format(
            self.budget, self.gt, self.est, self.lb, self.ub, self.true_error, self.lb_error, self.ub_error))
    
    def save(self, output_file: str, surfix: str = ""):
        output_file = output_file.split(".")[0] + surfix + ".jsonl"
        with open(f"{output_file}", "a+") as f:
            json.dump({
                "budget": int(self.budget),
                "est": float(self.est),
                "lb": float(self.lb),
                "ub": float(self.ub),
                "true_error": float(self.true_error),
                "lb_error": float(self.lb_error),
                "ub_error": float(self.ub_error)
                }, f)
            f.write("\n")
