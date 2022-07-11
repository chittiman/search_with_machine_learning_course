import fasttext 
from pathlib import Path

model_path = "/workspace/datasets/fasttext/title_model_normalized.bin"
in_file = "/workspace/datasets/fasttext/top_words.txt"
out_file = "/workspace/datasets/fasttext/synonyms.csv"

def load_file(file):
    with open(file,"r") as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_model(model_path):
    model = fasttext.load_model(model_path)
    return model

def get_synonyms(word,threshold =0.75):
    pairs = model.get_nearest_neighbors(word)
    synonyms = [pair[1] for pair in pairs if pair[0] > .75]
    return synonyms

def get_synonym_string(word,synonyms):
    all_words = [word] + synonyms
    synonym_string = ",".join(all_words) + "\n"
    return synonym_string

def write_synonym_file(in_file, out_file):
    out_lines = []
    words = load_file(in_file)
    for word in words:
        synonyms = get_synonyms(word)
        synonym_string = get_synonym_string(word,synonyms)
        out_lines.append(synonym_string)
    with open(out_file, "w+") as f:
        f.writelines(out_lines)


if __name__ == "__main__":
    model = load_model(model_path)
    write_synonym_file(in_file, out_file)


