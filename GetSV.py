import os
import json
import numpy as np
import pandas as pd

# Dimension list
dimensions = [
    "Realistic", "Deformation", "Imagination", "Color Richness",
    "Color Contrast", "Line Combination", "Line Texture",
    "Picture Organization", "Transformation"
]

# Load JSON file function
def load_json_data(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

# Calculate Score Volatility (SV)
def calculate_sv(score_data):
    user_scores = []
    
    for round_data in score_data:
        if round_data["round"] == 1:
            continue  # Skip round 1 initialization data
        # Get user score
        user_score = round_data['data']['scores']['current']
        if user_score is not None:
            user_scores.append(float(user_score))
    
    if len(user_scores) < 2:
        return np.nan  # Return NaN if not enough data

    # Calculate standard deviation to measure score volatility
    sv = np.std(user_scores)
    return sv

# Process each image and each dimension to calculate SV
def process_directory_for_sv(score_review_dir, output_file_sv):
    sv_results = []

    for image_num in range(1, 21):  # Adjust according to the actual number of images
        sv_values = []

        print(f"Processing image number: {image_num}")

        for dimension in dimensions:
            # Load score_Review file
            score_review_file = os.path.join(score_review_dir, f"{image_num}.jpg_{dimension}_score_Review.json")
            score_review_data = load_json_data(score_review_file)

            if score_review_data:
                print(f"Processing file: {score_review_file}")
                # Calculate score volatility
                sv_value = calculate_sv(score_review_data)
                sv_values.append(sv_value)

        # Calculate average SV for each image across dimensions
        avg_sv = np.nanmean(sv_values)

        print(f"Image {image_num}'s average SV: {avg_sv}")

        # Save results
        sv_results.append([f"{image_num}.jpg", avg_sv])

    # Save results to DataFrame and output to Excel file
    sv_df = pd.DataFrame(sv_results, columns=["File Name", "SV"])
    sv_df.to_excel(output_file_sv, index=False)
    print(f"SV results have been saved to: {output_file_sv}")

# Main function
if __name__ == "__main__":
    score_review_directory = "userActionsEveryRounds/score_Review"  # JSON file path
    output_file_sv = "SV_Results.xlsx"
    
    process_directory_for_sv(score_review_directory, output_file_sv)
