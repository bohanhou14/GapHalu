Pipeline to generate the detection datasets and run the detection experiments 

1. Generate prompts for the GPT4o-based detector:
```
python prompt_loader.py [train, test, small, large]
```

`train` and `test` modes each generates 10 examples of the four models.
`small` generates 100 per model.
`large` runs over the entire dataset.

2. Run the GPT4o detector via OpenAI API
```
python gpt_detect.py [train, test, small, large]
```
This runs detector over the instances and outputs a detection dataset (in HuggingFace format)

If you are replicating train/test results from the original paper, do:

3. Parse human annotations and add the labels to the detection dataset
```
python parse_human_annotations.py --detect_dataset DATASET_PATH
```
This outputs a DETECTION_HUMAN dataset with parsed human labels

4. Evaluate detection accuracy
```
python evaluate_detection.py DETECTION_HUMAN_DATASET_PATH
```
where DETECTION_HUMAN_DATASET_PATH is from step 3.

If you want to evaluate new generations, do:
```
python score_analysis.py DATASET_DIR
```
where DATASET_DIR contains a list of huggingface datasets to analyze the GapScore and GapHalu for.




