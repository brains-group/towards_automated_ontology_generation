import json
import argparse
import os


def process_results(input_path: str, output_path: str):
    # Load the JSON data
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Validate that 'results' exists and is a list
    if "results" not in data or not isinstance(data["results"], list):
        print("Error: Invalid JSON structure. Expected a 'results' array.")
        return

    results = data["results"]
    total_score = 0.0
    num_results = len(results)

    if num_results == 0:
        print("Warning: 'results' array is empty.")
        data["average_score"] = 0.0
    else:
        # Process each result
        for item in results:
            query_result = item.get("query_result", [])

            # If the query result is empty (e.g., an empty list []), set score to 0
            if not query_result:
                item["score"] = 0.0

            # Accumulate the score for the new average
            total_score += item.get("score", 0.0)

        # Update the average score without rounding
        new_average = total_score / num_results
        data["average_score"] = new_average

    # Create the output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save the updated dictionary to the output path
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

    print(f"Successfully processed {num_results} CQs.")
    print(f"Updated average score to: {data['average_score']}")
    print(f"Saved results to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero out scores for empty query results and recalculate average.")
    parser.add_argument("-i", "--input", required=True, help="Path to the input JSON file.")
    parser.add_argument("-o", "--output", required=True, help="Path to save the output JSON file.")

    args = parser.parse_args()

    process_results(args.input, args.output)