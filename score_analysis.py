from datasets import load_from_disk
import argparse
from collections import Counter
import os
import numpy as np
from rouge import Rouge
from tqdm import trange
import torch
import sys
from BARTScore.bart_score import BARTScorer
from loader import parse_generations_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse label lists from a text file")
    parser.add_argument("dataset_dir", help="directory containing the datasets to evaluate")
    args = parser.parse_args()
    gpt4o_generations = parse_generations_dataset()
    llama3_generations = parse_generations_dataset("clerc_generations/prompt1/Meta-Llama-3-8B-Instruct/preds")
    mistral_generations = parse_generations_dataset("clerc_generations/prompt1/Mistral-7B-Instruct-v0.3/preds")
    gemma_generations = parse_generations_dataset("clerc_generations/prompt1/gemma-1.1-7b-it/preds")
    file_list = os.listdir(args.dataset_dir)
    rouge = Rouge()
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    bart_scorer = BARTScorer(device=device, checkpoint='facebook/bart-large-cnn')
    # bart_scorer.load(path='bart.pth')
    gpt4o_rouge, llama3_rouge = [], []
    gpt4o_bart, llama3_bart = [], []
    gpt4o_halu, gpt4_nogap, gpt4_total = 0, 0, 0
    gpt4o_1, gpt4o_2, gpt4o_3 = 0, 0, 0
    gpt4_labels = []
    
    llama3_halu, llama3_nogap, llama3_total = 0, 0, 0
    llama3_1, llama3_2, llama3_3 = 0, 0, 0
    llama3_labels = []
    gpt4_failed, llama3_failed = 0, 0

    mistral_rouge, gemma_rouge = [], []
    mistral_bart, gemma_bart = [], []
    mistral_halu, mistral_nogap, mistral_total = 0, 0, 0
    mistral_1, mistral_2, mistral_3 = 0, 0, 0
    mistral_labels = []

    gemma_halu, gemma_nogap, gemma_total = 0, 0, 0
    gemma_1, gemma_2, gemma_3 = 0, 0, 0
    gemma_labels = []
    gemma_failed, mistral_failed = 0, 0

    for filename in file_list:
        ds = load_from_disk(os.path.join(args.dataset_dir, filename))
        labels = ds["labels"]
        docids = ds["docid"]
        data_ids = ds["data_id"]
        for idx in trange(len(labels), desc=f"Evaluating {filename}"):
            label = labels[idx]
            doc_id = docids[idx]
            data_id = data_ids[idx]

            # calculate rouge and bartscore
            if "gpt4o" in data_id:
                gpt4_total += 1
                if 0 in label and len(label) > 1:
                    gpt4_failed += 1
                    continue
                generation = [gen["generation"] for gen in gpt4o_generations if gen["docid"] == doc_id][0]
                gold_text = [gen["gold_text"] for gen in gpt4o_generations if gen["docid"] == doc_id][0]
            elif "llama3" in data_id:
                llama3_total += 1
                if 0 in label and len(label) > 1: # skip the erroneous examples
                    llama3_failed += 1
                    continue
                generation = [gen["generation"] for gen in llama3_generations if gen["docid"] == doc_id][0]
                gold_text = [gen["gold_text"] for gen in llama3_generations if gen["docid"] == doc_id][0]
            elif "mistral" in data_id:
                mistral_total += 1
                if 0 in label and len(label) > 1:
                    mistral_failed += 1
                    continue
                generation = [gen["generation"] for gen in mistral_generations if gen["docid"] == doc_id][0]
                gold_text = [gen["gold_text"] for gen in mistral_generations if gen["docid"] == doc_id][0]
            elif "gemma" in data_id:
                gemma_total += 1
                if 0 in label and len(label) > 1:
                    gemma_failed += 1
                    continue
                generation = [gen["generation"] for gen in gemma_generations if gen["docid"] == doc_id][0]
                gold_text = [gen["gold_text"] for gen in gemma_generations if gen["docid"] == doc_id][0]
            else:
                raise ValueError("Data ID does not contain a valid model name")
            try:
                rouge_score = rouge.get_scores(generation, gold_text)
                bart_score = bart_scorer.score([generation], [gold_text], batch_size=4)[0]
            except Exception as e:
                print(f"Error: {e}")
                continue
            
            # evaluate GapScore and GapHalu
            if "gpt4o" in data_id:
                gpt4o_rouge.append(rouge_score)
                gpt4o_bart.append(bart_score)
                if 0 in label:
                    gpt4_nogap += 1
                if 1 in label or 3 in label:
                    gpt4o_halu += 1
                
                if 1 in label:
                    gpt4o_1 += 1
                if 2 in label:
                    gpt4o_2 += 1
                if 3 in label:
                    gpt4o_3 += 1
                
                gpt4_labels.append(label)
                
            elif "llama3" in data_id:
                llama3_rouge.append(rouge_score)
                llama3_bart.append(bart_score)
                
                if 0 in label:
                    llama3_nogap += 1
                if 1 in label or 3 in label:
                    llama3_halu += 1
                
                if 1 in label:
                    llama3_1 += 1
                if 2 in label:
                    llama3_2 += 1
                if 3 in label:
                    llama3_3 += 1
                
                llama3_labels.append(label)
                
            elif "mistral" in data_id:
                mistral_rouge.append(rouge_score)
                mistral_bart.append(bart_score)

                if 0 in label:
                    mistral_nogap += 1
                if 1 in label or 3 in label:
                    mistral_halu += 1
                if 1 in label:
                    mistral_1 += 1
                if 2 in label:
                    mistral_2 += 1
                if 3 in label:
                    mistral_3 += 1
                
                mistral_labels.append(label)
            elif "gemma" in data_id:
                gemma_rouge.append(rouge_score)
                gemma_bart.append(bart_score)

                if 0 in label:
                    gemma_nogap += 1
                if 1 in label or 3 in label:
                    gemma_halu += 1
                if 1 in label:
                    gemma_1 += 1
                if 2 in label:
                    gemma_2 += 1
                if 3 in label:
                    gemma_3 += 1
                
                gemma_labels.append(label)

            else:
                raise ValueError("Data ID does not contain a valid model name")

            
    with open(f"analyses/results.txt", "w") as sys.stdout:
        print("Average ROUGE scores for GPT-4o:")
        print({k: sum([score[0][k]["f"] for score in gpt4o_rouge]) / len(gpt4o_rouge) for k in gpt4o_rouge[0][0].keys()})
        print("Average BART scores for GPT-4o:")
        print(np.mean(gpt4o_bart))

        print("Total number of examples for GPT-4o:", gpt4_total)
        print("GapScore for GPT-4o:")
        print(1 - (gpt4_nogap / gpt4_total))
        print("GapHalu for GPT-4o:")
        print(gpt4o_halu / gpt4_total)

        print("Gap Distributions for GPT-4o:")
        for k, v in Counter([tuple(label) for label in gpt4_labels]).items():
            print(f"Frequency of Gap {k} for GPT-4o:")
            print(v / gpt4_total)
            print("Value:", v)

        print("Frequency of Gap 1 for GPT-4o:")
        print(gpt4o_1 / gpt4_total)
        print("Frequency of Gap 2 for GPT-4o:")
        print(gpt4o_2 / gpt4_total)
        print("Frequency of Gap 3 for GPT-4o:")
        print(gpt4o_3 / gpt4_total)
        print("Total number of failed examples for GPT-4o:")
        print(gpt4_failed / gpt4_total)
        print("-" * 50)
        print("Unit testing:")
        print("GapHalu for GPT-4o", gpt4o_halu)
        print("G1 and G3", gpt4o_1 + gpt4o_3)
        print("G1 and G3 - GapHalu", gpt4o_1 + gpt4o_3 - gpt4o_halu)
        print("GapScore for GPT-4o", gpt4_total - gpt4_nogap)
        print("G1 and G2 and G3", gpt4o_1 + gpt4o_2 + gpt4o_3)

        print("-" * 50)

        print("Average ROUGE scores for Meta-Llama-3-8B-Instruct:")
        print({k: sum([score[0][k]["f"] for score in llama3_rouge]) / len(llama3_rouge) for k in llama3_rouge[0][0].keys()})
        print("Average BART scores for Meta-Llama-3-8B-Instruct:")
        print(np.mean(llama3_bart))

        print("Total number of examples for Meta-Llama-3-8B-Instruct:", llama3_total)
        print("GapScore for Meta-Llama-3-8B-Instruct:")
        print(1 - (llama3_nogap / llama3_total))
        print("GapHalu for Meta-Llama-3-8B-Instruct:")
        print(llama3_halu / llama3_total)

        
        print("Gap Distributions for Meta-Llama-3-8B-Instruct:")
        for k, v in Counter([tuple(label) for label in llama3_labels]).items():
            print(f"Frequency of Gap {k} for Meta-Llama-3-8B-Instruct:")
            print(v / llama3_total)
            print("Value:", v)
        print("Frequency of Gap 1 for Meta-Llama-3-8B-Instruct:")
        print(llama3_1 / llama3_total)
        print("Frequency of Gap 2 for Meta-Llama-3-8B-Instruct:")
        print(llama3_2 / llama3_total)
        print("Frequency of Gap 3 for Meta-Llama-3-8B-Instruct:")
        print(llama3_3 / llama3_total)
        print("Total number of failed examples for Meta-Llama-3-8B-Instruct:")
        print(llama3_failed / llama3_total)

        print("-" * 50)
        print("Unit testing:")
        print("GapHalu for Meta-Llama-3-8B-Instruct", llama3_halu)
        print("G1 and G3", llama3_1 + llama3_3)
        print("G1 and G3 - GapHalu", llama3_1 + llama3_3 - llama3_halu)
        print("GapScore for Meta-Llama-3-8B-Instruct", llama3_total - llama3_nogap)
        print("G1 and G2 and G3", llama3_1 + llama3_2 + llama3_3)
        print("-" * 50)

        print("Average ROUGE scores for Mistral-7B-Instruct-v0.3:")
        print({k: sum([score[0][k]["f"] for score in mistral_rouge]) / len(mistral_rouge) for k in mistral_rouge[0][0].keys()})
        print("Average BART scores for Mistral-7B-Instruct-v0.3:")
        print(np.mean(mistral_bart))

        print("Total number of examples for Mistral-7B-Instruct-v0.3:", mistral_total)
        print("GapScore for Mistral-7B-Instruct-v0.3:")
        print(1 - (mistral_nogap / mistral_total))
        print("GapHalu for Mistral-7B-Instruct-v0.3:")
        print(mistral_halu / mistral_total)

        print("Gap Distributions for Mistral-7B-Instruct-v0.3:")

        for k, v in Counter([tuple(label) for label in mistral_labels]).items():
            print(f"Frequency of Gap {k} for Mistral-7B-Instruct-v0.3:")
            print(v / mistral_total)
            print("Value:", v)
        print("Frequency of Gap 1 for Mistral-7B-Instruct-v0.3:")
        print(mistral_1 / mistral_total)
        print("Frequency of Gap 2 for Mistral-7B-Instruct-v0.3:")
        print(mistral_2 / mistral_total)
        print("Frequency of Gap 3 for Mistral-7B-Instruct-v0.3:")
        print(mistral_3 / mistral_total)
        print("Total number of failed examples for Mistral-7B-Instruct-v0.3:")
        print(mistral_failed / mistral_total)

        print("-" * 50)

        print("Average ROUGE scores for Gemma-1.1-7B-It:")
        print({k: sum([score[0][k]["f"] for score in gemma_rouge]) / len(gemma_rouge) for k in gemma_rouge[0][0].keys()})
        print("Average BART scores for Gemma-1.1-7B-It:")
        print(np.mean(gemma_bart))

        print("Total number of examples for Gemma-1.1-7B-It:", gemma_total)
        print("GapScore for Gemma-1.1-7B-It:")
        print(1 - (gemma_nogap / gemma_total))
        print("GapHalu for Gemma-1.1-7B-It:")
        print(gemma_halu / gemma_total)

        print("Gap Distributions for Gemma-1.1-7B-It:")
        for k, v in Counter([tuple(label) for label in gemma_labels]).items():
            print(f"Frequency of Gap {k} for Gemma-1.1-7B-It:")
            print(v / gemma_total)
            print("Value:", v)
        print("Frequency of Gap 1 for Gemma-1.1-7B-It:")
        print(gemma_1 / gemma_total)
        print("Frequency of Gap 2 for Gemma-1.1-7B-It:")
        print(gemma_2 / gemma_total)
        print("Frequency of Gap 3 for Gemma-1.1-7B-It:")
        print(gemma_3 / gemma_total)
        print("Total number of failed examples for Gemma-1.1-7B-It:")
        print(gemma_failed / gemma_total)





    
            



