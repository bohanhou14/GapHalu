from openai import OpenAI, AzureOpenAI
import os
import json
from argparse import ArgumentParser
import re
from tqdm import tqdm
from system_prompts.all_prompts import DETECT_SYSTEM_PROMPT_4, DETECT_SYSTEM_PROMPT_8, DETECT_SYSTEM_PROMPT_16, DETECT_SYSTEM_PROMPT_20
from datasets import Dataset

def init_openai_client(port=None, personal=False):
    openai_api_key = os.getenv("OPENAI_API_KEY") if not personal else os.getenv("OPENAI_API_KEY_PERSONAL")
    if port != None:
        openai_api_base = f"http://0.0.0.0:{port}/v1"
        return OpenAI(
            base_url=openai_api_base,
            api_key=openai_api_key
        )
    return OpenAI(
        api_key=openai_api_key
    )

def init_azure_openai_client():
    azure_openai_endpoint = os.getenv("AZURE_TX_ENDPOINT")
    azure_openai_api_key = os.getenv("AZURE_TX50k_KEY")
    return AzureOpenAI(
        api_key=azure_openai_api_key,
        api_version="2023-09-01-preview",
        azure_endpoint=azure_openai_endpoint
    )


def request_GPT(client, prompt, max_retries = 30, max_tokens=400, model="gpt-4o", system_prompt=None):
    num_try = 0
    # print(prompt)
    while num_try < max_retries:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens
            )
            if type(prompt) == list:
                return [r.text for r in response.choices]
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error: {e}")
            num_try += 1

if __name__ == "__main__":
    parser = ArgumentParser(description="Detect gaps in the generated text")
    parser.add_argument("mode", type=str, choices=['train', 'test', 'large', "small"], help="Total number of prompts to process")
    parser.add_argument("--model", default="gpt-4o", type=str, help="Model to use for detection")
    parser.add_argument("--port", type=int, help="Port to use for OpenAI API", default=8000)
    parser.add_argument("--batch_num", type=int, help="Batch number to process", default=0)
    parser.add_argument("--num_batches", type=int, help="Total number of batches", default=4)
    args = parser.parse_args()
    mode = args.mode

    # client = init_openai_client(personal=False)
    if args.model == "gpt-4o":
        client = init_azure_openai_client()
        model = os.getenv("AZURE_TX50k_DEV")
    else:
        client = init_openai_client(port=args.port)
        model = args.model
    
    prompt_list = [DETECT_SYSTEM_PROMPT_4, DETECT_SYSTEM_PROMPT_8, DETECT_SYSTEM_PROMPT_16, DETECT_SYSTEM_PROMPT_20]
    if mode == "train":
        dir_path = "prompts/train"
    elif mode == "test":
        dir_path = "prompts/test"
    elif mode == "large":
        dir_path = "prompts/large"
        prompt_list = [DETECT_SYSTEM_PROMPT_16]
    elif mode == "small": # detect mistral, gemma, and llama3, gpt4o with each 100 examples
        dir_path = "prompts/small"
        prompt_list = [DETECT_SYSTEM_PROMPT_16]
        
    file_list=os.listdir(dir_path)
    id_map = {}
    for filename in file_list:
        ids = filename.split(".")[0]
        data_id = ids.split("_")[0]
        docid = ids.split("_")[1]
        id_map[data_id] = docid

    # batch detect
    batch_lists = [file_list[i:i + len(file_list) // args.num_batches] for i in range(0, len(file_list), len(file_list) // args.num_batches)]
    batch = batch_lists[args.batch_num]
    
    prompt_num_map = {
        1: 4,
        2: 8,
        3: 16,
        4: 20
    }
    for prompt_idx, sys_prompt in enumerate(prompt_list):
        prompts = []
        ids = []
        docids = []
        detection_results = []
        exps = []
        for filename in tqdm(batch, desc=f"Detecting errors with num_shots={prompt_num_map[prompt_idx+1]}, on batch num {args.batch_num}", total=len(batch)):
            with open(os.path.join(dir_path, filename), "r") as f:
                prompt = f.read()
                response = ""
                while response == "":
                    response = request_GPT(client, prompt, model=model, system_prompt=sys_prompt)
                    try:
                        parsed_data = json.loads(response)
                        assert "label" in parsed_data and "explanation" in parsed_data, Exception("Failure to parse or missing fields in JSON")
                        data_id = filename.split(".")[0].split("_")[0]
                        ids.append(data_id)
                        docids.append(id_map[data_id])
                        prompts.append(prompt)
                        if type(parsed_data['label']) == list:
                            detection_results.append(parsed_data['label'])
                        else:
                            detection_results.append(list(parsed_data['label']))
                        exps.append(str(parsed_data['explanation']))
                    except Exception as e:
                        print(f"Failed to parse JSON: {e}")
                        response = ""

        dataset = Dataset.from_dict({
            "docid": docids,
            "data_id": ids,
            "prompt": prompts,
            "labels": detection_results,
            "explanations": exps
        })
        dataset.save_to_disk(f"outputs/{mode}-model={model}-num_shots={prompt_num_map[prompt_idx+1]}-batch={args.batch_num}")


    
    
    