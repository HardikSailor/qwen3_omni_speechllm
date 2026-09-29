import os
import evaluate


def compute_meteor(predictions=None, references=None, experiment_id=None):
    script_dir = os.path.dirname(os.path.realpath(__file__))
    meteor = evaluate.load(os.path.join(script_dir, "hf_metrics/meteor/meteor.py"), experiment_id=experiment_id)
    results = meteor.compute(predictions=predictions, references=references)

    return results["meteor"]


compute_meteor.__metric__ = "meteor"
compute_meteor.__greater_is_better__ = True
