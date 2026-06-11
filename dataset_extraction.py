
url = "https://raw.githubusercontent.com/Jordine/evaluation-awareness-probing/main/datasets/contrastive_dataset.json"

print(f"Fetching dataset from: {url}")

response = requests.get(url)
response.raise_for_status()

# 3. Parse the JSON
data = response.json()

# 4. Extract into aligned lists for the extraction loop
prompts = []
positive_targets = []
negative_targets = []

for entry in data:
    prompts.append(entry['question'])
    positive_targets.append(entry['positive'])
    negative_targets.append(entry['negative'])

print(f"\nSuccessfully loaded {len(prompts)} contrastive pairs.")
print("-" * 50)
print(f"Sample Prompt:\n{prompts[100]}")
print(f"Positive Target: {positive_targets[100]} | Negative Target: {negative_targets[100]}")
