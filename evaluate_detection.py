from datasets import load_from_disk
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import re
import os

def extract_integers_from_path(path):
    # Use regex to find all sequences of digits in the path
    integers = re.findall(r'\d+', path)
    
    # Convert the extracted digit sequences to integers
    integers = [int(num) for num in integers]
    assert len(integers) > 0, "No integers found in the path"

    return integers[-1]

def add_annotations(ax, bars):
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.2f}', 
                ha='center', va='bottom', fontsize=14, color='black')

if __name__ == "__main__":
    parser = ArgumentParser(description="Parse label lists from a text file")
    parser.add_argument("dataset_dir", help="directory containing the datasets to evaluate")
    parser.add_argument("--model", type=str, help="Model to use for detection, for naming the plot", default="gpt-4o")
    args = parser.parse_args()

    dataset_paths = []
    file_list = os.listdir(args.dataset_dir)
    for filename in file_list:
        if "human" in filename:
            dataset_paths.append(os.path.join(args.dataset_dir, filename))

    # sort dataset_path by int in descending order
    dataset_paths = sorted(dataset_paths, key=extract_integers_from_path)
    exs, ps, rs, f1s = [], [], [], []
    
    for ds_path in dataset_paths:
        over_l1, over_l2, over_l3 = 0, 0, 0
        under_l1, under_l2, under_l3 = 0, 0, 0
        total_l1, total_l2, total_l3 = 0, 0, 0
        dataset = load_from_disk(ds_path)
        human_labels = dataset["human_labels"]
        labels = dataset["labels"]
        exact_match_total, exact_match_inaccuracy = 0, 0
        precisions, recalls, f1_scores = [], [], []
        for i, (human_label, label) in enumerate(zip(human_labels, labels)):
            if len(human_label) == 0:
                continue
            label_set = set(label)
            human_label_set = set(human_label)
            exact_match_total += 1 
            exact_match_inaccuracy += 1 if label_set != human_label_set else 0
            intersection = label_set.intersection(human_label_set)
            if 1 in label_set and 1 not in human_label_set:
                over_l1 += 1
            elif 1 not in label_set and 1 in human_label_set:
                under_l1 += 1
            if 2 in label_set and 2 not in human_label_set:
                over_l2 += 1
            elif 2 not in label_set and 2 in human_label_set:
                under_l2 += 1
            if 3 in label_set and 3 not in human_label_set:
                over_l3 += 1
            elif 3 not in label_set and 3 in human_label_set:
                under_l3 += 1
            if 1 in human_label_set:
                total_l1 += 1
            if 2 in human_label_set:
                total_l2 += 1
            if 3 in human_label_set:
                total_l3 += 1

            precisions.append(len(intersection) / len(label_set))
            recalls.append(len(intersection) / len(human_label_set))
            if precisions[-1] + recalls[-1] == 0:
                f1_scores.append(0)
            else:
                f1_scores.append(2 * precisions[-1] * recalls[-1] / (precisions[-1] + recalls[-1]))
            
        exact_match_accuracy = 1 - exact_match_inaccuracy / exact_match_total
        exs.append(exact_match_accuracy)
        ps.append(sum(precisions) / len(precisions))
        rs.append(sum(recalls) / len(recalls))
        f1s.append(sum(f1_scores) / len(f1_scores))
        print(f"Dataset: {ds_path}")
        print(f"Over L: [{over_l1 / total_l1}, {over_l2 / total_l2}, {over_l3 / total_l3}]")
        print(f"Under L: [{under_l1 / total_l1}, {under_l2 / total_l2}, {under_l3 / total_l3}]")
        

    
    
    shots_nums = [extract_integers_from_path(dataset_path) for dataset_path in dataset_paths]
    
    plot_x = ["#" + str(num) + "s" for num in shots_nums]

    plt.rcParams.update({
        'font.size': 16,          # Default font size for all text
        'axes.titlesize': 20,     # Font size for axes titles
        'axes.labelsize': 20,     # Font size for axes labels
        'xtick.labelsize': 15,    # Font size for x-axis tick labels
        'ytick.labelsize': 14,    # Font size for y-axis tick labels
        'legend.fontsize': 12,    # Font size for legend text
        'figure.titlesize': 24    # Font size for figure title
    })
    fig, axes = plt.subplots(2,2, figsize=(10, 8))
    colors = ['#8FDD8F', '#8FBC8F',  '#008080', '#4F7942']
    bars = axes[0, 0].bar(plot_x, exs, color=colors, label='Exact Match Accuracy')
    axes[0,0].set_ylabel("mG-ExactMatch")
    axes[0,0].set_ylim(0, 1)
    axes[0,0].set_title("mG-ExactMatch vs # of Shots")
    add_annotations(axes[0, 0], bars)

    bars = axes[0,1].bar(plot_x, ps, color=colors, label='Precision')
    axes[0,1].set_ylabel("mG-Precision")
    axes[0,1].set_ylim(0, 1)
    axes[0,1].set_title("mG-Precision vs # of Shots")
    add_annotations(axes[0, 1], bars)

    bars = axes[1,0].bar(plot_x, rs, color=colors, label='Recall')
    axes[1,0].set_ylabel("mG-Recall")
    axes[1,0].set_ylim(0, 1)
    axes[1,0].set_title("mG-Recall vs # of Shots")
    add_annotations(axes[1, 0], bars)

    bars = axes[1,1].bar(plot_x, f1s, color=colors, label='F1 Score')
    axes[1,1].set_ylabel("mG-F1")
    axes[1,1].set_ylim(0, 1)
    axes[1,1].set_title("mG-F1 vs # of Shots")
    add_annotations(axes[1, 1], bars)
    fig.suptitle(f"Metrics for {args.model}")

    plt.tight_layout()
    plt.savefig(f"{args.model}-metrics.png")


    