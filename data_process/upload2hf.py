from huggingface_hub import login
from datasets import Dataset
import pandas as pd
import json

login(token="xxx")

# dataset = json.load(open('gsm8k.json'))
# dataset = json.load(open('math.json'))
# dataset = json.load(open('math2.json'))
# dataset = json.load(open('math-zeroshot.json'))
# dataset = json.load(open('math-zeroshot-chat.json'))
# dataset = json.load(open('gsm8k-chat.json'))
dataset = json.load(open('ultrafeedback_first_5000.json'))

# Create a simple dataset
# dataset = {
#     "context_messages": dataset
# }
df = pd.DataFrame(dataset)
dataset = Dataset.from_pandas(df)

# dataset.push_to_hub("PeterJinGo/gsm8k", private=False)
# dataset.push_to_hub("PeterJinGo/math", private=False)
# dataset.push_to_hub("PeterJinGo/math2", private=False)
# dataset.push_to_hub("PeterJinGo/math-zeroshot", private=False)
# dataset.push_to_hub("PeterJinGo/math-zeroshot-chat", private=False)
# dataset.push_to_hub("PeterJinGo/gsm8k-chat", private=False)
dataset.push_to_hub("PeterJinGo/ultrafeedback_first_5000", private=False)
