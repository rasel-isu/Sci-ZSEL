import json

def mean_reciprocal_rank(predictions, ground_truths):
    # create_data()
    assert len(predictions) == len(ground_truths), "Mismatch in number of queries"

    reciprocal_ranks = []

    for pred_list, true_id in zip(predictions, ground_truths):
        try:
            rank = pred_list.index(true_id) + 1  # ranks start at 1
            reciprocal_ranks.append(1.0 / rank)
        except ValueError:
            reciprocal_ranks.append(0.0)  # true_id not found in prediction

    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
    return mrr
# Example data: predictions are ranked lists
# predictions = [
#     ['A', 'B', 'C'],   # Correct is 'B' → rank 2 → 1/2
#     ['D', 'E', 'F'],   # Correct is 'F' → rank 3 → 1/3
#     ['G', 'H', 'I'],   # Correct is 'Z' (not in list) → 0
# ]

# # Ground truth for each query
# ground_truths = ['B', 'F', 'Z']

# dataset_file_path = "crossencoder_predictions_grag.json"
# dataset_file_path = "Dataset/crossencoder_predictions_grag.json"
# dataset_file_path = "dataset_ho/crossencoder_predictions_grag.json"
#create data

def process_data_for_mrr(dataset_file_path):
    predictions = []
    ground_truths = []

    # with open(dataset_file_path, 'r', encoding='utf-8') as f:
    #     data = json.load(f)
    with open(dataset_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
        for i, entry in enumerate(data):

            print(f"\nProcessing entry {i+1}:")
            
            # Extract ground truth
            ground_truth_id = entry['ground_truth']['id']
            ground_truth_title = entry['ground_truth']['title']
            ground_truths.append(ground_truth_id)
            print(f"  Ground Truth: {ground_truth_id} - {ground_truth_title}")
            
            # Extract predictions
            predicted_ids = [candidate['id'] for candidate in entry['retrieved_candidates']]
            predictions.append(predicted_ids)
            print(f"  Total candidates: {len(predicted_ids)}")
    # return predictions, ground_truths
    mrr_score = mean_reciprocal_rank(predictions, ground_truths)

    with open(dataset_file_path.replace('.json', '_mrr_result.json'), 'w') as f:
        f.write(f"MRR: {mrr_score:.4f}\n")


process_data_for_mrr("dataset_ho/crossencoder_predictions_grag.json")



# predictions, ground_truths = create_data(dataset_file_path)

# Compute MRR
# mrr_score = mean_reciprocal_rank(predictions, ground_truths)
# print(f"MRR: {mrr_score:.4f}")

# output_file = "mrr_result_ho_prime.txt"
# with open(output_file, "w") as f:
#     f.write(f"MRR: {mrr_score:.4f}\n")
# print(f"MRR score saved to {output_file}")

