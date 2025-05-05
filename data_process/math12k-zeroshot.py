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
train_solution = []
train_answer = []
cnt = 0
for d in tqdm(dataset):
    if not re.search(pattern, d['solution']):
        cnt += 1
        continue
            
    train_data.append([{
        'content': prompt_template.format(d['problem']),
        'role': 'user'
    }])
        
    train_solution.append(d['solution'])
    train_answer.append(re.search(pattern, d['solution']).group(1))

print(cnt)
json.dump({'context_messages': train_data, 'answer': train_answer, 'solution': train_solution}, open('math-zeroshot.json', 'w'), indent=4)
