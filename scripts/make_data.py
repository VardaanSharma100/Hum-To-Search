from datasets import load_dataset

print("Starting download....")

dataset = load_dataset("amanteur/CHAD_hummings")

print("Download complete.")
print(dataset)