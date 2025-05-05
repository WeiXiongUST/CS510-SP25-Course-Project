from datasets import load_dataset
import json
from IPython import embed
from tqdm import tqdm

dataset = load_dataset('openai/gsm8k', 'main')['train']

prompt_template = "Your task is to answer the question below. Give step by step reasoning before you answer,\nand when you're ready to answer, please use the format 'Final answer: ...'\nQuestion: {}\nSolution:"

train_data = []
train_answer = []
for d in tqdm(dataset):
    train_data.append([{
        'content': prompt_template.format(d['question']),
        'role': 'user'
    }])
    train_answer.append(d['answer'])

json.dump({'context_messages': train_data, 'answer': train_answer}, open('gsm8k.json', 'w'), indent=4)
