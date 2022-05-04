import os
from flair.data import Sentence
from pathlib import Path

def get_o_for_empty_strings(s: str):
    if s == "":
        return "O"
    return s


encoding = "utf-8"
text_files_folder = "../text_files/"
text_files_directory = os.fsencode(text_files_folder)

Path("../flair_style_training_data").mkdir(parents=True, exist_ok=True)
save_file = open("../flair_style_training_data/all.txt", "w", encoding=encoding)

annotation_information = {}
annotation_file = open("../subtrack1_entities/distemist_subtrack1_training_mentions.tsv", "r", encoding=encoding)
for line in annotation_file:
    row_split = line.strip().split("\t")
    filename = row_split[0]  # will be used as identifier in dict
    mark = row_split[1]
    label = row_split[2]
    off0 = row_split[3]
    off1 = row_split[4]
    span = row_split[5]  # text is at index 5 of tsv
    if span == "span":  # only in case of tsv header
        continue
    text_split = span.split(" ")  # contains single words of text column
    information_object = {
        "filename": filename,
        "mark": mark,
        "label": label,
        "off0": off0,
        "off1": off1,
        "span": span
    }
    if filename in annotation_information:
        annotation_information[filename].append(information_object)
    else:
        annotation_information[filename] = [information_object]

files_amount = len(os.listdir(text_files_directory))
files_parsed = 0
amount_of_files_to_parse = 2

for file in os.listdir(text_files_directory):
    file_name = os.fsdecode(file)
    if ".txt" in file_name:
        files_parsed += 1

        text_file = open(text_files_folder + file_name, "r", encoding=encoding)
        text_file_identifier = file_name[:-4]
        text = text_file.read()
        sentence = Sentence(text)

        for tkn in sentence:
            for annotation in annotation_information[text_file_identifier]:
                offset_start = int(annotation["off0"])
                offset_end = int(annotation["off1"])
                if tkn.start_position == offset_start:
                    tkn.add_label("ner", f"B-ENFERMEDAD")
                elif tkn.start_position > offset_start and tkn.end_position <= offset_end:
                    tkn.add_label("ner", f"I-ENFERMEDAD")

        # write annotated sentence into a training file
        save_file.writelines([f'{tkn.text} {get_o_for_empty_strings(tkn.get_label("ner").value)}\n' for tkn in sentence] + ["\n"])

        if files_parsed % 100 == 0:
            print(f" ... {(files_parsed*100/files_amount):6.2f}% parsed ({files_parsed} out of {files_amount})")
        if amount_of_files_to_parse == files_parsed:
            save_file.close()
            quit()

save_file.close()
