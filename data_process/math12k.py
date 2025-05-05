from datasets import load_dataset
import json
from IPython import embed
from tqdm import tqdm
# from random import sample
import random
import re

random.seed(2024)
pattern = r"\\boxed\{((?:[^\{\}]+|\{(?:[^\{\}]+|\{[^\{\}]*\})*\})*)\}"


# def random_few_shot(dataset, d):
#     filtered_dataset = dataset.filter(lambda row: row != d, disable=True)
#     filtered_rows = list(filtered_dataset)
#     assert len(filtered_rows) + 1 == len(dataset)
#     selected_rows = sample(filtered_rows, 4)
    
#     return selected_rows

def random_few_shot(dataset, d, n=4):
    # Convert dataset to a list for shuffling (if not already a list)
    dataset_list = list(dataset)
    random.shuffle(dataset_list)  # Randomly shuffle the dataset
    
    selected_rows = []
    for row in dataset_list:
        if row != d and re.search(pattern, row['solution']):
            selected_rows.append(row)
            if len(selected_rows) == n:  # Stop once we have enough rows
                break
    
    return selected_rows

dataset = load_dataset('lighteval/MATH', 'all')['train']

prompt_template = """Your task is to answer the last question below. Give step by step reasoning before you answer, and when you're ready to answer, please wrap your answer in \\boxed
Question: {}
Solution: {}
Question: {}
Solution: {}
Question: {}
Solution: {}
Question: {}
Solution: {}
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
    
    few_shot = random_few_shot(dataset, d)
    
    box_contents = [re.search(pattern, sample['solution']).group(1) for sample in few_shot]
        
    train_data.append([{
        'content': prompt_template.format(few_shot[0]['problem'],
                                          few_shot[0]['solution'],
                                          few_shot[1]['problem'],
                                          few_shot[1]['solution'],
                                          few_shot[2]['problem'],
                                          few_shot[2]['solution'],
                                          few_shot[3]['problem'],
                                          few_shot[3]['solution'],
                                          d['problem']),
        'role': 'user'
    }])
        
    train_solution.append(d['solution'])
    train_answer.append(re.search(pattern, d['solution']).group(1))

print(cnt)
json.dump({'context_messages': train_data, 'answer': train_answer, 'solution': train_solution}, open('math.json', 'w'), indent=4)
