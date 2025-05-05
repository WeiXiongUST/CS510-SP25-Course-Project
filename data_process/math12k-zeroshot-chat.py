from datasets import load_dataset
import json
from IPython import embed
from tqdm import tqdm
# from random import sample
import random
import re

random.seed(2024)
pattern = r"\\boxed\{((?:[^\{\}]+|\{(?:[^\{\}]+|\{[^\{\}]*\})*\})*)\}"


dataset = load_dataset('lighteval/MATH', 'all')['train']

prompt_template = """Your task is to answer the last question below. Give step by step reasoning before you answer, and when you're ready to answer, please wrap your answer in \\boxed
Question: {}
Solution: """

train_data = []
cnt = 0
for d in tqdm(dataset):
    if not re.search(pattern, d['solution']):
        cnt += 1
        continue
            
    train_data.append([{'content': prompt_template.format(d['problem']),'role': 'user'},
                       {'content': d['solution'],'role': 'assistant'}])

print(cnt)
json.dump({'messages': train_data}, open('math-zeroshot-chat.json', 'w'), indent=4)
