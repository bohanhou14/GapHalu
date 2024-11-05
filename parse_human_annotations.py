import re
import os
import argparse
from datasets import load_from_disk, Dataset

def parse_labels(text):
    # Define regex pattern to match "Label:" followed by one or more integers, possibly separated by commas
    pattern = r"Label:\s*([\d\s,]+)"
    
    # Find all matches for the pattern in the text
    matches = re.findall(pattern, text)
    
    # Parse and collect all label values as a list of lists of integers
    parsed_labels = []
    for match in matches:
        # Split the matched string by commas to get individual numbers
        numbers_str = match
        
        # Ensure that only valid integers are captured (strip whitespace and check if digit)
        numbers = [num.strip() for num in numbers_str.split(',') if num.strip().isdigit()]
        
        # Convert each valid string number to an integer
        integers_list = list(map(int, numbers))
        parsed_labels.append(integers_list)
    
    return parsed_labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse label lists from a text file")
    parser.add_argument("parse_dir", help="parse human annotations from the assigned directory")
    parser.add_argument("--detect_dataset", help="detect dataset to add human annotations to")
    args = parser.parse_args()

    # parse human annotations
    path = args.parse_dir
    example_labels = []
    ones, twos, threes, zeros = 0, 0, 0, 0
    example_ids = []
    file_list = os.listdir(path)
    sorted_file_list = sorted(file_list, key=lambda x: int(x.split('-')[1].split('.')[0]))
    for idx, filename in enumerate(sorted_file_list):
        with open(os.path.join(path, filename), "r") as f:
            lines = f.readlines()
            text = " ".join(lines[1:])
            text.replace("\\n", "")
            parsed_lists = parse_labels(text)
            if len(parsed_lists) > 0:
                example_labels.append(parsed_lists[0])
                example_ids.append(lines[0].strip())
                print(f"Processed {filename} with labels: {parsed_lists[0]}")
                print(f"Example ID: {lines[0].strip()}")
                if 1 in parsed_lists[0]:
                    ones += 1 
                if 2 in parsed_lists[0]:
                    twos += 1
                if 3 in parsed_lists[0]:
                    threes += 1
                if 0 in parsed_lists[0]:
                    zeros += 1
    print(f"Total number of examples: {ones + twos + threes + zeros}")
    total = ones + twos + threes + zeros
    print(f"Percentage of 1s: {ones / total * 100}")
    print(f"Percentage of 2s: {twos / total * 100}")
    print(f"Percentage of 3s: {threes / total * 100}")
    print(f"Percentage of 0s: {zeros / total * 100}")
    # calculate the shannon entropy and the balance
    from numpy import log, sum
    def shannon_entropy(probs):
        return -sum([p * log(p) if p != 0 else 0 for p in probs]) 
    ent = shannon_entropy([ones / total, twos / total, threes / total, zeros / total])
    print(f"Shannon entropy: {ent}")
    print(f"standard_ent = {shannon_entropy([0.25, 0.25, 0.25, 0.25])}")
    print(f"Balance: {ent / log(4)}")



                
    if args.detect_dataset:
        dataset = load_from_disk(args.detect_dataset)
        ids = dataset["data_id"]
        prompt = dataset["prompt"]
        labels = dataset["labels"]
        explanations = dataset["explanations"]
        human_labels = []
        for idx, data_id in enumerate(ids):
            if data_id in example_ids:
                human_labels.append(example_labels[example_ids.index(data_id)])
            else:
                human_labels.append([])
            
        data = {
            "data_id": ids,
            "prompt": prompt,
            "labels": labels,
            "explanations": explanations,
            "human_labels": human_labels
        }
        dataset = Dataset.from_dict(data)
        dataset.save_to_disk(args.detect_dataset + "_human")