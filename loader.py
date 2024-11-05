'''
This file loads the datasets and converts the data into a format that is friendly to human annotators.
'''

import json
from datasets import load_dataset
import sys
import textwrap
import os

def parse_generations_dataset(dir_path="clerc_generations/prompt1/gpt-4o/preds"):
    generations = []
    for filename in os.listdir(dir_path):
        with open(os.path.join(dir_path, filename), "r") as f:
            # load json
            data = json.load(f)
            # get docid
            new_data = {
                "docid": data["meta"]["docid"],
                "generation": data["gen"],
                "gold_text": data["meta"]["gold_text"]
            }
            generations.append(new_data)
    return generations

def wrap_text(text, width=100):
    return textwrap.fill(text, width=width)

if __name__ == "__main__":
    dataset = load_dataset("jhu-clsp/CLERC", data_files={"data": f"generation/test.jsonl"})["data"]
    # train_dataset = load_dataset("jhu-clsp/CLERC", data_files={"data": f"generation/train.jsonl"})["data"]
    gpt4o_generations = parse_generations_dataset()
    llama3_generations = parse_generations_dataset("clerc_generations/prompt1/Meta-Llama-3-8B-Instruct/preds")
    llama3ft_generations = parse_generations_dataset("clerc_generations/prompt1/llama3-ft/preds")
    
    written = 0; idx = 0; cutoff = 100
    while written < cutoff:
        docid = dataset[idx]["docid"]
        gold_text = dataset[idx]["gold_text"]
        prev_text = dataset[idx]["previous_text"]
        citations = [cite[0] for cite in dataset[idx]["citations"]]
        short_citations = dataset[idx]["short_citations"]
        # breakpoint()
        gpt4o_generation = [gen["generation"] for gen in gpt4o_generations if gen["docid"] == docid]
        llama3_generation = [gen["generation"] for gen in llama3_generations if gen["docid"] == docid]
        llama3ft_generation = [gen["generation"] for gen in llama3ft_generations if gen["docid"] == docid]
        if len(gpt4o_generation) == 0 or len(llama3_generation) == 0 or len(llama3ft_generation) == 0:
            idx += 1
            continue
        
        with open(f"examples/test-{written+1}.md", "w") as f:
            f.write("**gold_text:**\n")
            f.write(wrap_text(gold_text) + "\n\n")
            f.write("**Generations:**\n\n")
            f.write("\n***GPT-4o:***\n")
            f.write(wrap_text(gpt4o_generation[0]) + "\n\n") 
            f.write("\n**Label:** \n\n")
            f.write("\n***Meta-Llama-3-8B-Instruct:***\n")
            f.write(wrap_text(llama3_generation[0]) + "\n\n")
            f.write("\n**Label:** \n\n")
            f.write("\n***Meta-Llama-3-8B-Instruct-finetuned:***\n")
            f.write(wrap_text(llama3ft_generation[0]) + "\n\n")
            f.write("\n**Label:** \n\n")
            f.write("\n**citations:** " + wrap_text(str(citations)) + "\n\n")
            for i, cite in enumerate(short_citations):
                f.write(f"***short_citations_{i}:*** " + cite + "\n\n")
            f.write("**prev_text:**\n")
            f.write(wrap_text(prev_text) + "\n\n")
            f.write("\n")
            f.close()
        idx += 1
        written += 1

# with open("examples/test-1.txt", "w") as f:
#     f.write(json.dumps(dataset[0], indent=4) + "\n")
