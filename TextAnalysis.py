import os
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Define the normalize function
def normalize(value, min_value, max_value):
    if max_value - min_value == 0:
        return 0
    return (value - min_value) / (max_value - min_value)


# Define the 9 dimension names
dimensions = [
    "Realistic", "Deformation", "Imagination", "Color Richness",
    "Color Contrast", "Line Combination", "Line Texture",
    "Picture Organization", "Transformation"
]


# Load JSON file
def load_json_data(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None


# Calculate TAR (Text Acceptance Rate)
def get_tar(text_data):
    total_tar = 0
    valid_rounds = 0  # Record valid rounds

    for round_data in text_data:
        if not isinstance(round_data, dict):
            continue  # Ensure round_data is a dictionary
        if round_data.get("round") == 1:
            print("Skipping round 1 data")
            continue  # Skip round 1

        # Process the 'reviews' field
        reviews_data = round_data.get('data', {}).get('Reviews', None)
        if reviews_data:
            original = reviews_data.get('original', "")
            added = reviews_data.get('added', "")
            removed = reviews_data.get('removed', "")

            if original:
                len_original = len(original)
                len_added = len(added)
                len_removed = len(removed)

                denominator = len_added + len_original
                if denominator > 0:
                    tar_round = (len_original - len_removed) / denominator
                    total_tar += tar_round
                    valid_rounds += 1

        # Process the 'suggestions' field
        suggestions_data = round_data.get('data', {}).get('suggestions', None)
        if suggestions_data:
            original = suggestions_data.get('original', "")
            added = suggestions_data.get('added', "")
            removed = suggestions_data.get('removed', "")

            if original:
                len_original = len(original)
                len_added = len(added)
                len_removed = len(removed)

                denominator = len_added + len_original
                if denominator > 0:
                    tar_round = (len_original - len_removed) / denominator
                    total_tar += tar_round
                    valid_rounds += 1

    if valid_rounds == 0:
        print("No valid rounds")
        return np.nan  # Return NaN if no valid rounds

    # Calculate the average TAR
    average_tar = total_tar / valid_rounds
    return average_tar


# Calculate TS (Text Similarity)
def get_ts(text_data):
    gpt_texts = []
    user_texts = []

    for round_data in text_data:
        if round_data["round"] == 1:
            continue

        # Process the 'reviews' field
        reviews_data = round_data['data'].get('Reviews', {})
        if reviews_data:
            gpt_text = reviews_data.get('original', "")
            user_text = reviews_data.get('current', "")
            if gpt_text.strip() and user_text.strip():
                gpt_texts.append(gpt_text)
                user_texts.append(user_text)

        # Process the 'suggestions' field
        suggestions_data = round_data['data'].get('suggestions', {})
        if suggestions_data:
            gpt_text = suggestions_data.get('original', "")
            user_text = suggestions_data.get('current', "")
            if gpt_text.strip() and user_text.strip():
                gpt_texts.append(gpt_text)
                user_texts.append(user_text)

    if not gpt_texts or not user_texts:
        return np.nan  # Return NaN if no data

    # Use bag-of-words model to analyze based on words, not characters
    vectorizer = CountVectorizer(analyzer='word', token_pattern=r"(?u)\b\w+\b").fit([gpt_texts[-1], user_texts[-1]])

    # Vectorize the GPT and user texts
    gpt_vector = vectorizer.transform([gpt_texts[-1]]).toarray()
    user_vector = vectorizer.transform([user_texts[-1]]).toarray()

    if gpt_vector.shape[1] > 1 and user_vector.shape[1] > 1:
        similarity = cosine_similarity(gpt_vector, user_vector)[0][0]
        return similarity  # Return the unnormalized cosine similarity
    return np.nan  # Return NaN if no data


# Process directory and calculate TAR and TS
def process_directory(score_comment_dir, suggestion_dir, output_file_tar, output_file_ts):
    tar_results = []
    ts_results = []

    for image_num in range(1, 21):  # Adjust based on actual range
        tar_values = []
        ts_values = []

        for dimension in dimensions:
            # Process score_Review file
            score_comment_file = os.path.join(score_comment_dir, f"{image_num}.jpg_{dimension}_score_Review.json")
            score_comment_data = load_json_data(score_comment_file)

            if score_comment_data:
                # Calculate TAR and TS for reviews
                tar_value_review = get_tar(score_comment_data)
                ts_value_review = get_ts(score_comment_data)

                # Store the review results
                tar_values.append(tar_value_review)
                ts_values.append(ts_value_review)

            # Process suggestions file
            suggestion_file = os.path.join(suggestion_dir, f"{image_num}.jpg_{dimension}_suggestion.json")
            suggestion_data = load_json_data(suggestion_file)

            if suggestion_data:
                # Calculate TAR and TS for suggestions
                tar_value_suggestion = get_tar(suggestion_data)
                ts_value_suggestion = get_ts(suggestion_data)

                # Store the suggestion results
                tar_values.append(tar_value_suggestion)
                ts_values.append(ts_value_suggestion)

        # Append results for TAR and TS
        tar_results.append([f"{image_num}.jpg"] + tar_values)
        ts_results.append([f"{image_num}.jpg"] + ts_values)

    # Define columns for Review and Suggestion TAR/TS for each dimension
    tar_columns = ["File Name"] + [f"{dim}_Review_TAR" for dim in dimensions] + [f"{dim}_Suggestion_TAR" for dim in
                                                                                 dimensions]
    ts_columns = ["File Name"] + [f"{dim}_Review_TS" for dim in dimensions] + [f"{dim}_Suggestion_TS" for dim in
                                                                               dimensions]

    # Save TAR results to Excel
    tar_df = pd.DataFrame(tar_results, columns=tar_columns)
    tar_df.to_excel(output_file_tar, index=False)
    print(f"TAR results have been saved to: {output_file_tar}")

    # Save TS results to Excel
    ts_df = pd.DataFrame(ts_results, columns=ts_columns)
    ts_df.to_excel(output_file_ts, index=False)
    print(f"TS results have been saved to: {output_file_ts}")


# Main function
if __name__ == "__main__":
    score_comment_directory = "userActionsEveryRounds/score_Review"  # Path to score review JSON files
    suggestion_directory = "userActionsEveryRounds/suggestion"  # Path to suggestion JSON files
    output_file_tar = "TAR_Results.xlsx"
    output_file_ts = "TS_Results.xlsx"

    process_directory(score_comment_directory, suggestion_directory, output_file_tar, output_file_ts)
