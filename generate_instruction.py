"""
batch_selfinstruct_generate.py

run:
python -m generate_instruction generate_instruction_following_data \
  --output_dir ./ \
  --num_instructions_to_generate 10 \
  --model_name="text-davinci-003" \
  
python -m generate_instruction generate_instruction_following_data_ja \
    --output_dir ./ \
    --num_instructions_to_generate 10 \
    --model_name="mistralai/Mistral-7B-v0.1" \
"""
import time
import json
import os
import random
import re
import string
from functools import partial
from multiprocessing import Pool

import numpy as np
import tqdm
from rouge_score import rouge_scorer
from rouge_score.tokenizers import Tokenizer
import utils

import fire
import MeCab
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class MeCabTokenizer(Tokenizer):
    """rouge-score用のMeCabを用いたTokenizerクラス

    Args:
        use_stemmer (bool, optional): Trueの場合、単語の原型で分割する. Defaults to False.
    """
    def __init__(self, use_stemmer=False):
        self._stemmer = use_stemmer
        
        self.tagger = MeCab.Tagger()
        self.wakati = MeCab.Tagger("-Owakati")


    def tokenize(self, text):
        if self._stemmer:
            node = self.tagger.parseToNode(text)
            original_forms = []
            while node:
                feature = node.feature.split(",")
                original_forms.append(feature[6])
                node = node.next

            return original_forms
        
        else:
            return self.wakati.parse(text).split()


def encode_prompt(prompt_instructions):
    """Encode multiple prompt instructions into a single string."""
    prompt = open("./prompt.txt").read() + "\n"

    for idx, task_dict in enumerate(prompt_instructions):
        (instruction, input, output) = task_dict["instruction"], task_dict["input"], task_dict["output"]
        instruction = re.sub(r"\s+", " ", instruction).strip().rstrip(":")
        input = "<noinput>" if input.lower() == "" else input
        prompt += f"###\n"
        prompt += f"{idx + 1}. Instruction: {instruction}\n"
        prompt += f"{idx + 1}. Input:\n{input}\n"
        prompt += f"{idx + 1}. Output:\n{output}\n"
    prompt += f"###\n"
    prompt += f"{idx + 2}. Instruction:"
    return prompt


def encode_prompt_ja(prompt_instructions):
    """Encode multiple prompt instructions into a single string."""
    prompt = open("./prompt_ja.txt").read() + "\n"

    for idx, task_dict in enumerate(prompt_instructions):
        (instruction, input, output) = task_dict["instruction"], task_dict["input"], task_dict["output"]
        instruction = re.sub(r"\s+", " ", instruction).strip().rstrip(":")
        input = "<noinput>" if input.lower() == "" else input
        prompt += f"###\n"
        prompt += f"{idx + 1}. Instruction: {instruction}\n"
        prompt += f"{idx + 1}. Input:\n{input}\n"
        prompt += f"{idx + 1}. Output:\n{output}\n"
    prompt += f"###\n"
    prompt += f"{idx + 2}. Instruction:"
    return prompt


def post_process_gpt3_response(num_prompt_instructions, response):
    if response is None:
        return []
    raw_instructions = f"{num_prompt_instructions+1}. Instruction:" + response["text"]
    raw_instructions = re.split("###", raw_instructions)
    instructions = []
    for idx, inst in enumerate(raw_instructions):
        # if the decoding stops due to length, the last example is likely truncated so we discard it
        if idx == len(raw_instructions) - 1 and response["finish_reason"] == "length":
            continue
        idx += num_prompt_instructions + 1
        splitted_data = re.split(f"{idx}\.\s+(Instruction|Input|Output):", inst)
        if len(splitted_data) != 7:
            continue
        else:
            inst = splitted_data[2].strip()
            input = splitted_data[4].strip()
            input = "" if input.lower() == "<noinput>" else input
            output = splitted_data[6].strip()
        # filter out too short or too long instructions
        if len(inst.split()) <= 3 or len(inst.split()) > 150:
            continue
        # filter based on keywords that are not suitable for language models.
        blacklist = [
            "image",
            "images",
            "graph",
            "graphs",
            "picture",
            "pictures",
            "file",
            "files",
            "map",
            "maps",
            "draw",
            "plot",
            "go to",
            "video",
            "audio",
            "music",
            "flowchart",
            "diagram",
        ]
        blacklist += []
        if any(find_word_in_string(word, inst) for word in blacklist):
            continue
        # We found that the model tends to add "write a program" to some existing instructions, which lead to a lot of such instructions.
        # And it's a bit comfusing whether the model need to write a program or directly output the result.
        # Here we filter them out.
        # Note this is not a comprehensive filtering for all programming instructions.
        if inst.startswith("Write a program"):
            continue
        # filter those starting with punctuation
        if inst[0] in string.punctuation:
            continue
        # filter those starting with non-english character
        if not inst[0].isascii():
            continue
        instructions.append({"instruction": inst, "input": input, "output": output})
    return instructions


def post_process_hf_transformers_response(num_prompt_instructions, result):
    if result is None:
        return []
    raw_instructions = f"{num_prompt_instructions+1}. Instruction:" + result
    raw_instructions = re.split("###", raw_instructions)
    instructions = []
    for idx, inst in enumerate(raw_instructions):
        # 最後の例は長さの問題で止まる可能性が高いので破棄する
        if idx == len(raw_instructions) - 1:
            continue
        # サンプル数+1からスタートしている
        idx += num_prompt_instructions + 1
        splitted_data = re.split(f"{idx}\.\s+(Instruction|Input|Output):", inst)
        # 分割数で正しく生成されたか確認する
        if len(splitted_data) != 7:
            continue
        else:
            inst = splitted_data[2].strip()
            input = splitted_data[4].strip()
            input = "" if input.lower() == "<noinput>" else input
            output = splitted_data[6].strip()
        # 短すぎたり長すぎたりする指示を除外する
        if len(inst) <= 10 or len(inst) > 300:
            continue
        # 言語モデルに適さないキーワードを含むものをフィルタリングする
        blacklist = [
            "画像",
            "絵画",
            "グラフ",
            "図",
            "写真",
            "絵",
            "ファイル",
            "地図",
            "図表",
            "図面",
            "に行って",
            "動画",
            "映像",
            "音声",
            "音楽",
            "フローチャート",
            "図表",
        ]
        blacklist += []
        if any(contains_keyword(inst, word) for word in blacklist):
            continue
        # 私たちは、このモデルが既存のいくつかの命令に「プログラムを書く」を追加する傾向があることを発見した。
        # そして、モデルがプログラムを書く必要があるのか、結果を直接出力する必要があるのか、ちょっと混乱する。
        # ここでは、それらをフィルタリングする。
        # これは、すべてのプログラミング指示に対する包括的なフィルタリングではないことに注意する。
        if inst.startswith("プログラムを書く"):
            continue
        # 句読点で始まるものをフィルタする
        if inst[0] in {",", ".", "!", "?", "、", "。", "！", "？"}:
            continue
        # 英語以外の文字で始まるものをフィルタリングする
        # if not inst[0].isascii():
        #     continue
        instructions.append({"instruction": inst, "input": input, "output": output})
    return instructions


def find_word_in_string(w, s):
    return re.compile(r"\b({0})\b".format(w), flags=re.IGNORECASE).search(s)


def contains_keyword(text, keyword):
    return keyword in text


def generate_instruction_following_data(
    output_dir="./",
    seed_tasks_path="./seed_tasks.jsonl",
    num_instructions_to_generate=100,
    model_name="text-davinci-003",
    num_prompt_instructions=3,
    request_batch_size=5,
    temperature=1.0,
    top_p=1.0,
    num_cpus=16,
):
    seed_tasks = [json.loads(l) for l in open(seed_tasks_path, "r")]
    seed_instruction_data = [
        {"instruction": t["instruction"], "input": t["instances"][0]["input"], "output": t["instances"][0]["output"]}
        for t in seed_tasks
    ]
    print(f"Loaded {len(seed_instruction_data)} human-written seed instructions")

    os.makedirs(output_dir, exist_ok=True)
    request_idx = 0
    # load the LM-generated instructions
    machine_instruction_data = []
    if os.path.exists(os.path.join(output_dir, "regen.json")):
        machine_instruction_data = utils.jload(os.path.join(output_dir, "regen.json"))
        print(f"Loaded {len(machine_instruction_data)} machine-generated instructions")

    # similarities = {}
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    # now let's generate new instructions!
    progress_bar = tqdm.tqdm(total=num_instructions_to_generate)
    if machine_instruction_data:
        progress_bar.update(len(machine_instruction_data))

    # first we tokenize all the seed instructions and generated machine instructions
    all_instructions = [d["instruction"] for d in seed_instruction_data] + [
        d["instruction"] for d in machine_instruction_data
    ]
    all_instruction_tokens = [scorer._tokenizer.tokenize(inst) for inst in all_instructions]

    while len(machine_instruction_data) < num_instructions_to_generate:
        request_idx += 1

        batch_inputs = []
        for _ in range(request_batch_size):
            # only sampling from the seed tasks
            prompt_instructions = random.sample(seed_instruction_data, num_prompt_instructions)
            prompt = encode_prompt(prompt_instructions)
            batch_inputs.append(prompt)
        decoding_args = utils.OpenAIDecodingArguments(
            temperature=temperature,
            n=1,
            max_tokens=3072,  # hard-code to maximize the length. the requests will be automatically adjusted
            top_p=top_p,
            stop=["\n20", "20.", "20."],
        )
        request_start = time.time()
        results = utils.openai_completion(
            prompts=batch_inputs,
            model_name=model_name,
            batch_size=request_batch_size,
            decoding_args=decoding_args,
            logit_bias={"50256": -100},  # prevent the <|endoftext|> token from being generated
        )
        request_duration = time.time() - request_start

        process_start = time.time()
        instruction_data = []
        for result in results:
            new_instructions = post_process_gpt3_response(num_prompt_instructions, result)
            instruction_data += new_instructions

        total = len(instruction_data)
        keep = 0
        for instruction_data_entry in instruction_data:
            # computing similarity with the pre-tokenzied instructions
            new_instruction_tokens = scorer._tokenizer.tokenize(instruction_data_entry["instruction"])
            with Pool(num_cpus) as p:
                rouge_scores = p.map(
                    partial(rouge_scorer._score_lcs, new_instruction_tokens),
                    all_instruction_tokens,
                )
            rouge_scores = [score.fmeasure for score in rouge_scores]
            most_similar_instructions = {
                all_instructions[i]: rouge_scores[i] for i in np.argsort(rouge_scores)[-10:][::-1]
            }
            if max(rouge_scores) > 0.7:
                continue
            else:
                keep += 1
            instruction_data_entry["most_similar_instructions"] = most_similar_instructions
            instruction_data_entry["avg_similarity_score"] = float(np.mean(rouge_scores))
            machine_instruction_data.append(instruction_data_entry)
            all_instructions.append(instruction_data_entry["instruction"])
            all_instruction_tokens.append(new_instruction_tokens)
            progress_bar.update(1)
        process_duration = time.time() - process_start
        print(f"Request {request_idx} took {request_duration:.2f}s, processing took {process_duration:.2f}s")
        print(f"Generated {total} instructions, kept {keep} instructions")
        utils.jdump(machine_instruction_data, os.path.join(output_dir, "regen.json"))


def generate_instruction_following_data_ja(
    output_dir="./",
    seed_tasks_path="./seed_tasks_ja.jsonl",
    num_instructions_to_generate=100,
    model_name="mistralai/Mixtral-8x22B-v0.1",
    num_prompt_instructions=3,
    temperature=0.7,
    top_p=0.95,
    top_k=40,
    max_new_tokens=4096,
    num_cpus=4,
):
    seed_tasks = [json.loads(l) for l in open(seed_tasks_path, "r")]
    seed_instruction_data = [
        {"instruction": t["instruction"], "input": t["instances"][0]["input"], "output": t["instances"][0]["output"]}
        for t in seed_tasks
    ]
    print(f"Loaded {len(seed_instruction_data)} human-written seed instructions")

    os.makedirs(output_dir, exist_ok=True)
    request_idx = 0
    # 言語モデルによって生成された指示を読み込む
    machine_instruction_data = []
    if os.path.exists(os.path.join(output_dir, "regen_ja.json")):
        machine_instruction_data = utils.jload(os.path.join(output_dir, "regen_ja.json"))
        print(f"Loaded {len(machine_instruction_data)} machine-generated instructions")

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False, tokenizer=MeCabTokenizer())

    # さて、新しい指示を生成しましょう！
    progress_bar = tqdm.tqdm(total=num_instructions_to_generate)
    if machine_instruction_data:
        progress_bar.update(len(machine_instruction_data))

    # 最初に、すべてのシード指示と生成された機械によって生成された指示をトークン化します
    all_instructions = [d["instruction"] for d in seed_instruction_data] + [
        d["instruction"] for d in machine_instruction_data
    ]
    all_instruction_tokens = [scorer._tokenizer.tokenize(inst) for inst in all_instructions]

    # モデルとトークナイザを読み込む
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=False,
    )

    while len(machine_instruction_data) < num_instructions_to_generate:
        request_idx += 1

        # シードタスクからのみサンプリング
        prompt_instructions = random.sample(seed_instruction_data, num_prompt_instructions)
        prompt = encode_prompt_ja(prompt_instructions)

        request_start = time.time()
        result = utils.hf_transformers_completion(
            prompt,
            model,
            tokenizer,
            temperature=temperature,
            do_sample=True,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
        )
        request_duration = time.time() - request_start

        process_start = time.time()
        instruction_data = post_process_hf_transformers_response(num_prompt_instructions, result)

        total = len(instruction_data)
        keep = 0
        for instruction_data_entry in instruction_data:
            # トークナイズされた指示との類似度を計算
            new_instruction_tokens = scorer._tokenizer.tokenize(instruction_data_entry["instruction"])
            with Pool(num_cpus) as p:
                rouge_scores = p.map(
                    partial(rouge_scorer._score_lcs, new_instruction_tokens),
                    all_instruction_tokens,
                )
            rouge_scores = [score.fmeasure for score in rouge_scores]
            most_similar_instructions = {
                all_instructions[i]: rouge_scores[i] for i in np.argsort(rouge_scores)[-10:][::-1]
            }
            if max(rouge_scores) > 0.7:
                continue
            else:
                keep += 1
            instruction_data_entry["most_similar_instructions"] = most_similar_instructions
            instruction_data_entry["avg_similarity_score"] = float(np.mean(rouge_scores))
            machine_instruction_data.append(instruction_data_entry)
            all_instructions.append(instruction_data_entry["instruction"])
            all_instruction_tokens.append(new_instruction_tokens)
            progress_bar.update(1)
        process_duration = time.time() - process_start
        print(f"Request {request_idx} took {request_duration:.2f}s, processing took {process_duration:.2f}s")
        print(f"Generated {total} instructions, kept {keep} instructions")
        utils.jdump(machine_instruction_data, os.path.join(output_dir, "regen_ja.json"))


def main(task, **kwargs):
    globals()[task](**kwargs)


if __name__ == "__main__":
    fire.Fire(main)
