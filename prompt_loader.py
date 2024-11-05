'''
This file helps load the dataset and parse the relevant data into the prompt designed for the detection task.
'''
from datasets import load_dataset
from loader import parse_generations_dataset
from argparse import ArgumentParser

if __name__ == "__main__":
    dataset = load_dataset("jhu-clsp/CLERC", data_files={"data": f"generation/test.jsonl"})["data"]
    gpt4o_generations = parse_generations_dataset()
    llama3_generations = parse_generations_dataset("clerc_generations/prompt1/Meta-Llama-3-8B-Instruct/preds")
    mistral_generations = parse_generations_dataset("clerc_generations/prompt1/Mistral-7B-Instruct-v0.3/preds")
    gemma_generations = parse_generations_dataset("clerc_generations/prompt1/gemma-1.1-7b-it/preds")
    parser = ArgumentParser(description="Load the dataset and generate prompts for the detection task")
    parser.add_argument("mode", type=str, choices=["train", "test", "large", "small"])
    args = parser.parse_args()
    
    if args.mode == "train":
        start = 0
        end = 10
        out_dir = "prompts/train"
    elif args.mode == "test":
        start = 10
        end = 20
        out_dir = "prompts/test"
    elif args.mode == 'large':
        start = 10
        end = len(dataset)
        out_dir = "prompts/large"
    elif args.mode == 'small':
        start = 10
        end = 110
        out_dir = "prompts/small"
    else:
        raise NotImplementedError("The mode is not implemented")
    
    for idx in range(start, end):
        data = dataset[idx]
        docid = data["docid"]
        gold_text = data["gold_text"]
        prev_text = data["previous_text"]
        short_citations = data["short_citations"]
        citations = [cite[0] for cite in data["citations"]]
        gpt4o_generation = [gen["generation"] for gen in gpt4o_generations if gen["docid"] == docid]
        llama3_generation = [gen["generation"] for gen in llama3_generations if gen["docid"] == docid]
        mistral_generation = [gen["generation"] for gen in mistral_generations if gen["docid"] == docid]
        gemma_generation = [gen["generation"] for gen in gemma_generations if gen["docid"] == docid]
        if len(gpt4o_generation) == 0 or len(llama3_generation) == 0:
            continue

        prompt = '''Output a valid JSON object with the fields of {"label": [(one or more integers from 0-3 indicating the gap categories, expressed in a list)], "explanation": a short explanation justifying the label.}. Do not output anything else such as 'json' or newline characters or redundant spaces. Answer after output: '''
        
        with open(f"{out_dir}/gpt4o-{idx+1}_{docid}.txt", "w") as f:
            f.write("Generation:\n\n")
            f.write(gpt4o_generation[0] + "\n\n")
            f.write("citations needed to make: " + str(citations) + "\n\n")
            f.write("target_text: " + gold_text + "\n\n")
            for i, cite in enumerate(short_citations):
                f.write(f"reference_case_{i+1}: " + cite + "\n\n")
            f.write("previous_text: " + prev_text + "\n\n")
            f.write(prompt + "\n\n")
            f.write("output: ")
            f.close()
        
        with open(f"{out_dir}/llama3-{idx+1}_{docid}.txt", "w") as f:
            f.write("**Generation:**\n\n")
            f.write(llama3_generation[0] + "\n\n")
            f.write("citations needed to make: " + str(citations) + "\n\n")
            f.write("gold_text: " + gold_text + "\n\n")
            for i, cite in enumerate(short_citations):
                f.write(f"reference_case_{i+1}: " + cite + "\n\n")
            f.write("previous_text: " + prev_text + "\n\n")
            f.write(prompt + "\n\n")
            f.write("output: ")
            f.close()
        
        with open(f"{out_dir}/mistral-{idx+1}_{docid}.txt", "w") as f:
            f.write("**Generation:**\n\n")
            f.write(mistral_generation[0] + "\n\n")
            f.write("citations needed to make: " + str(citations) + "\n\n")
            f.write("gold_text: " + gold_text + "\n\n")
            for i, cite in enumerate(short_citations):
                f.write(f"reference_case_{i+1}: " + cite + "\n\n")
            f.write("previous_text: " + prev_text + "\n\n")
            f.write(prompt + "\n\n")
            f.write("output: ")
            f.close()
        
        with open(f"{out_dir}/gemma-{idx+1}_{docid}.txt", "w") as f:
            f.write("**Generation:**\n\n")
            f.write(gemma_generation[0] + "\n\n")
            f.write("citations needed to make: " + str(citations) + "\n\n")
            f.write("gold_text: " + gold_text + "\n\n")
            for i, cite in enumerate(short_citations):
                f.write(f"reference_case_{i+1}: " + cite + "\n\n")
            f.write("previous_text: " + prev_text + "\n\n")
            f.write(prompt + "\n\n")
            f.write("output: ")
            f.close()


            

            
    