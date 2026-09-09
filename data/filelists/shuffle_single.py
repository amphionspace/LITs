import random

train_val_ratio = 0.9

with open("./en_data_9017_subset.txt", "r", encoding="utf-8") as f:
    en_lines = [line.strip() for line in f if line.strip()]


random.seed(42)
random.shuffle(en_lines)

train_list = en_lines[:int(len(en_lines) * train_val_ratio)]
val_list = en_lines[int(len(en_lines) * train_val_ratio):]
assert len(train_list) + len(val_list) == len(en_lines), "Train and validation lists do not sum up to total lines."

with open("./en_data_9017_subset_training.txt", "w", encoding="utf-8") as f:
    for line in train_list:
        f.write(line + "\n")

with open("./en_data_9017_subset_validation.txt", "w", encoding="utf-8") as f:
    for line in val_list:
        f.write(line + "\n")

print("Train set size:", len(train_list))
print("Validation set size:", len(val_list))