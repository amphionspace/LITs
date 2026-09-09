cleaned = []

with open("./en_9017_and_short_22k.txt", "r") as f:
    for line in f:
        parts = line.split("|")
        text_with_newline = parts[-1]
        cleaned.append(text_with_newline)

with open("english_training_text.txt", "w") as f:
    f.writelines(cleaned)