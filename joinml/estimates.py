import logging
import json


class Estimates:
    def __init__(self, cost: float, gt: float, est: float, lbs, ubs) -> None:
        self.gt = gt
        self.est = est
        self.lbs = lbs if isinstance(lbs, list) else [lbs]
        self.ubs = ubs if isinstance(ubs, list) else [ubs]
        self.true_error = (self.est - self.gt) / self.gt
        self.lb_errors = [(lb - self.gt) / self.gt for lb in self.lbs]
        self.ub_errors = [(ub - self.gt) / self.gt for ub in self.ubs]
        self.cost = cost
        self.coverages = [int(self.gt >= lb and self.gt <= ub) for lb, ub in zip(self.lbs, self.ubs)]

    def log(self):
        logging.info("budget: {} true_error {} lb_errors {} ub_errors {} coverages {}".format(
            self.cost, self.true_error, self.lb_errors, self.ub_errors, self.coverages))
    
    def save(self, output_file: str, surfix: str = ""):
        output_file = output_file.rsplit(".", 1)[0] + surfix + ".jsonl"
        with open(f"{output_file}", "a+") as f:
            json.dump({
                "budget": float(self.cost),
                "gt": float(self.gt),
                "est": float(self.est),
                "true_error": float(self.true_error),
                "lbs": [float(lb) for lb in self.lbs],
                "ubs": [float(ub) for ub in self.ubs],
                "lb_errors": [float(lb_error) for lb_error in self.lb_errors],
                "ub_errors": [float(ub_error) for ub_error in self.ub_errors],
                "coverages": [int(coverage) for coverage in self.coverages]
                }, f)
            f.write("\n")

class Selection:
    def __init__(self, cost: float, type: str, target: float, recall: float, precision: float, status: bool) -> None:
        self.cost = cost
        self.type = type
        self.target = target
        self.recall = recall
        self.precision = precision
        self.status = status
    
    def log(self):
        logging.info("budget: {} type {} target {} recall {} precision {}".format(
            self.cost, self.type, self.target, self.recall, self.precision))
        
    def save(self, output_file: str, surfix: str = ""):
        output_file = output_file.rsplit(".", 1)[0] + surfix + ".jsonl"
        with open(f"{output_file}", "a+") as f:
            json.dump({
                "budget": float(self.cost),
                "type": self.type,
                "target": float(self.target),
                "recall": float(self.recall),
                "precision": float(self.precision),
                "status": str(self.status)
                }, f)
            f.write("\n")

class TopK:
    def __init__(self, cost: float, gt: set, est: set) -> None:
        self.gt = gt
        self.est = est
        self.precision = len(self.gt.intersection(self.est)) / len(self.est) if len(self.est) > 0 else 0.0
        self.recall = len(self.gt.intersection(self.est)) / len(self.gt) if len(self.gt) > 0 else 0.0
        self.cost = cost
    
    def log(self):
        logging.info("budget: {} gt {} est {} precision {} recall {}".format(
            self.cost, self.gt, self.est, self.precision, self.recall))
    
    def save(self, output_file: str, surfix: str = ""):
        output_file = output_file.rsplit(".", 1)[0] + surfix + ".jsonl"
        with open(f"{output_file}", "a+") as f:
            json.dump({
                "budget": float(self.cost),
                "gt": list(self.gt),
                "est": list(self.est),
                "precision": float(self.precision),
                "recall": float(self.recall)
                }, f)
            f.write("\n")