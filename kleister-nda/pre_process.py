import argparse
import json
import os
from PIL import Image
from transformers import AutoTokenizer


def bbox_string(box, width, length):
    return (
        str(int(1000 * (box[0] / width)))
        + " "
        + str(int(1000 * (box[1] / length)))
        + " "
        + str(int(1000 * (box[2] / width)))
        + " "
        + str(int(1000 * (box[3] / length)))
    )


def actual_bbox_string(box, width, length):
    return (
        str(box[0])
        + " "
        + str(box[1])
        + " "
        + str(box[2])
        + " "
        + str(box[3])
        + "\t"
        + str(width)
        + " "
        + str(length)
    )


def convert(data_dir, output_dir, data_split):
    with open(
        os.path.join(output_dir, data_split + ".txt.tmp"),
        "w",
        encoding="utf8",
    ) as fw, open(
        os.path.join(output_dir, data_split + "_box.txt.tmp"),
        "w",
        encoding="utf8",
    ) as fbw, open(
        os.path.join(output_dir, data_split + "_image.txt.tmp"),
        "w",
        encoding="utf8",
    ) as fiw:
        for file in os.listdir(data_dir):
            file_path = os.path.join(data_dir, file)
            with open(file_path, "r", encoding="utf8") as f:
                data = json.load(f)
            image_path = file_path.replace("annotations", "images")
            img = data['form'][0]['image']
            image_path = image_path.replace(file, img)
            file_name = os.path.basename(image_path)
            image = Image.open(image_path)
            width, length = image.size
            for item in data["form"]:
                words, label = item["words"], item["label"]
                words = [w for w in words if w["text"].strip() != ""]
                if len(words) == 0:
                    continue
                if label == "other":
                    for w in words:
                        fw.write(w["text"] + "\tO\n")
                        fbw.write(
                            w["text"]
                            + "\t"
                            + bbox_string(w["box"], width, length)
                            + "\n"
                        )
                        fiw.write(
                            w["text"]
                            + "\t"
                            + actual_bbox_string(w["box"], width, length)
                            + "\t"
                            + file_name
                            + "\n"
                        )
                else:
                    if len(words) == 1:
                        fw.write(words[0]["text"] + "\tS-" + label.upper() + "\n")
                        fbw.write(
                            words[0]["text"]
                            + "\t"
                            + bbox_string(words[0]["box"], width, length)
                            + "\n"
                        )
                        fiw.write(
                            words[0]["text"]
                            + "\t"
                            + actual_bbox_string(words[0]["box"], width, length)
                            + "\t"
                            + file_name
                            + "\n"
                        )
                    else:
                        fw.write(words[0]["text"] + "\tB-" + label.upper() + "\n")
                        fbw.write(
                            words[0]["text"]
                            + "\t"
                            + bbox_string(words[0]["box"], width, length)
                            + "\n"
                        )
                        fiw.write(
                            words[0]["text"]
                            + "\t"
                            + actual_bbox_string(words[0]["box"], width, length)
                            + "\t"
                            + file_name
                            + "\n"
                        )
                        for w in words[1:-1]:
                            fw.write(w["text"] + "\tI-" + label.upper() + "\n")
                            fbw.write(
                                w["text"]
                                + "\t"
                                + bbox_string(w["box"], width, length)
                                + "\n"
                            )
                            fiw.write(
                                w["text"]
                                + "\t"
                                + actual_bbox_string(w["box"], width, length)
                                + "\t"
                                + file_name
                                + "\n"
                            )
                        fw.write(words[-1]["text"] + "\tE-" + label.upper() + "\n")
                        fbw.write(
                            words[-1]["text"]
                            + "\t"
                            + bbox_string(words[-1]["box"], width, length)
                            + "\n"
                        )
                        fiw.write(
                            words[-1]["text"]
                            + "\t"
                            + actual_bbox_string(words[-1]["box"], width, length)
                            + "\t"
                            + file_name
                            + "\n"
                        )
            fw.write("\n")
            fbw.write("\n")
            fiw.write("\n")


def seg_file(file_path, tokenizer, max_len):
    subword_len_counter = 0
    output_path = file_path[:-4]
    with open(file_path, "r", encoding="utf8") as f_p, open(
        output_path, "w", encoding="utf8"
    ) as fw_p:
        for line in f_p:
            line = line.rstrip()

            if not line:
                fw_p.write(line + "\n")
                subword_len_counter = 0
                continue
            token = line.split("\t")[0]

            current_subwords_len = len(tokenizer.tokenize(token))

            # Token contains strange control characters like \x96 or \x95
            # Just filter out the complete line
            if current_subwords_len == 0:
                continue

            if (subword_len_counter + current_subwords_len) > max_len:
                fw_p.write("\n" + line + "\n")
                subword_len_counter = current_subwords_len
                continue

            subword_len_counter += current_subwords_len

            fw_p.write(line + "\n")


def seg(model_name_or_path, output_dir, data_split, max_len):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, do_lower_case=True
    )
    seg_file(
        os.path.join(output_dir, data_split + ".txt.tmp"),
        tokenizer,
        max_len,
    )
    seg_file(
        os.path.join(output_dir, data_split + "_box.txt.tmp"),
        tokenizer,
        max_len,
    )
    seg_file(
        os.path.join(output_dir, data_split + "_image.txt.tmp"),
        tokenizer,
        max_len,
    )


def main(data_dir, data_split, output_dir, model_name_or_path, max_len):
    #convert(data_dir, output_dir, data_split)
    seg(model_name_or_path, output_dir, data_split, max_len)
