from datasets import load_dataset, Dataset

# Load the dataset
d = load_dataset('RLHFlow/ultrafeedback_iter1')

# Extract the first 5000 data points
first_5000 = d['train'].select(range(5000))

# Save the subset to disk in JSON format
first_5000.to_json('ultrafeedback_first_5000.json')


from huggingface_hub import HfApi

api = HfApi()

# Create a new dataset repo (replace 'your-username/first-5000-dataset' with your Hugging Face ID)
repo_id = "PeterJinGo/ultrafeedback_first_5000"
api.create_repo(repo_id, repo_type="dataset")

# Upload the file to the created dataset repository
api.upload_file(
    path_or_fileobj="ultrafeedback_first_5000.json",
    path_in_repo="ultrafeedback_first_5000.json",
    repo_id=repo_id,
    repo_type="dataset"
)

print(f"Dataset uploaded to: https://huggingface.co/datasets/{repo_id}")
