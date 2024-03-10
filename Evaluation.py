import random

import json
# with open('new_train.json', 'rt') as f:

with open('queries_train.json', 'rt') as f:
  queries = json.load(f)
def average_precision(true_list, predicted_list, k=40):
    true_set = frozenset(true_list)
    predicted_list = predicted_list[:k]
    precisions = []
    for i,doc_id in enumerate(predicted_list):
        # if int(doc_id) in true_set:

        if doc_id in true_set:
            prec = (len(precisions)+1) / (i+1)
            precisions.append(prec)
    if len(precisions) == 0:
        return 0.0
    return round(sum(precisions)/len(precisions),3)
def precision_at_k(true_list, predicted_list, k):
    true_set = frozenset(true_list)
    predicted_list = predicted_list[:k]
    if len(predicted_list) == 0:
        return 0.0
    return round(len([1 for doc_id in predicted_list if doc_id in true_set]) / len(predicted_list), 3)
def recall_at_k(true_list, predicted_list, k):
    true_set = frozenset(true_list)
    predicted_list = predicted_list[:k]
    if len(true_set) < 1:
        return 1.0
    return round(len([1 for doc_id in predicted_list if doc_id in true_set]) / len(true_set), 3)
def f1_at_k(true_list, predicted_list, k):
    p = precision_at_k(true_list, predicted_list, k)
    # print(f"Patk: {p}")
    r = recall_at_k(true_list, predicted_list, k)
    # print(f"recallatk: {r}")
    if p == 0.0 or r == 0.0:
        return 0.0
    return round(2.0 / (1.0/p + 1.0/r), 3)
def results_quality(true_list, predicted_list):
    p5 = precision_at_k(true_list, predicted_list, 5)
    # print(f"P5: {p5}")
    f1_30 = f1_at_k(true_list, predicted_list, 30)
    # print(f"F1: {f1_30}")
    if p5 == 0.0 or f1_30 == 0.0:
        return 0.0
    return round(2.0 / (1.0/p5 + 1.0/f1_30), 3)

# Initialize weights
pr_weight = 0.7
pv_weight = 0.3

# Parameters for the optimization process
learning_rate = 0.05
num_iterations = 100
tolerance = 0.01  # Minimum improvement to continue optimization

best_quality = 0.0
best_weights = (pr_weight, pv_weight)
import requests
from time import time
url = 'http://34.170.46.156:8080'
def calculate_gradient(quality_function, pr_weight, pv_weight, epsilon=1e-4):
    """
    Approximate gradient calculation for the quality function with respect to both weights.
    """
    grad_pr_weight = (quality_function(pr_weight + epsilon, pv_weight) - quality_function(pr_weight, pv_weight)) / epsilon
    grad_pv_weight = (quality_function(pr_weight, pv_weight + epsilon) - quality_function(pr_weight, pv_weight)) / epsilon
    return grad_pr_weight, grad_pv_weight


def evaluate_quality(pr_weight, pv_weight):
    current_quality = 0
    for q, true_ids in queries.items():
        params = {
            'query': q,
            'pv_weight': pv_weight,
            'pr_weight': pr_weight
        }
        try:
            res = requests.get(url + '/search', params=params, timeout=35)
            res.raise_for_status()  # This will raise an exception for HTTP errors

            # Attempt to decode JSON only if response is successful
            try:
                predicted_ids, _ = zip(*res.json())
                quality = results_quality(true_ids, predicted_ids)
                current_quality += quality
            except json.JSONDecodeError:
                print(f"Failed to decode JSON from response for query: {q}")
                continue  # Skip this iteration and move to the next query

        except requests.RequestException as e:
            print(f"Request failed for query: {q}, error: {e}")
            continue  # Skip this iteration and move to the next query

    if len(queries) > 0:
        return current_quality / len(queries)
    else:
        return 0



for iteration in range(num_iterations):
    current_quality = evaluate_quality(pr_weight, pv_weight)
    grad_pr_weight, grad_pv_weight = calculate_gradient(evaluate_quality, pr_weight, pv_weight)

    # Update weights based on gradient
    pr_weight -= learning_rate * grad_pr_weight
    pv_weight -= learning_rate * grad_pv_weight

    # Normalize weights to ensure they sum to 1 and remain within [0, 1]
    total_weight = pr_weight + pv_weight
    if total_weight > 0:
        pr_weight, pv_weight = pr_weight / total_weight, pv_weight / total_weight
    else:
        pr_weight, pv_weight = 0.5, 0.5

    # Re-check quality after weight adjustment
    new_quality = evaluate_quality(pr_weight, pv_weight)

    # Check for improvement
    if new_quality > best_quality + tolerance:
        best_quality = new_quality
        best_weights = (pr_weight, pv_weight)
    else:
        # Revert to previous best weights if no improvement
        pr_weight, pv_weight = best_weights

    print(f"Iteration {iteration + 1}: Quality = {new_quality}, Weights = (pr_weight: {pr_weight}, pv_weight: {pv_weight})")

# Final optimized weights
print(f"Optimized Weights: Text = {best_weights[0]}, Title = {best_weights[1]}")




