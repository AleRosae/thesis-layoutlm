# LayoutLM for Visual Information Extraction (VIE)
This repository contains the code that I have wrote while carrying out experiments for my final thesis for the MA of Digital Humanities and Digital Knowledge at the University of Bologna.

Experiments were carried out on two different public datasets: FUNSD and Kleister-NDA. The two folders contain all the code necessary to reproduce the experiments, along with the training logs that contain the final scores of the fine-tuning process. 

The fine-tuned models were uploaded on the HuggingFace Hub and can be found at the following links:

LayoutLMv1 - FUNSD: https://huggingface.co/Sennodipoi/LayoutLMv1-FUNSD-ft

LayoutLMv2 - FUNSD: https://huggingface.co/Sennodipoi/LayoutLMv2-FUNSD-ft
LayoutLMv3 - FUNSD: https://huggingface.co/Sennodipoi/LayoutLMv3-FUNSD-ft

LayoutLMv1 - Kleister-NDA: https://huggingface.co/Sennodipoi/LayoutLMv1-kleisterNDA
LayoutLMv2 - Kleister-NDA: https://huggingface.co/Sennodipoi/LayoutLMv2-kleisterNDA
LayoutLMv3 - Kleister-NDA: https://huggingface.co/Sennodipoi/LayoutLMv3-kleisterNDA

# General Pipeline
FUNSD does not require any additional pre-processing steps and can be fine-tuned following the great tutorials already available from the HuggingFace community (https://github.com/NielsRogge/Transformers-Tutorials).

Kleister-NDA, on the other hand, requires additional pre-processing to convert the raw .pdf files into a format that is readable by LayoutLM. 

`create_annotations.py`: this script maps the labels that were provided by the original authors in .tsv files into the text extracted from OCR. Notice that documents were processed using an internal software so it cannot work directly as it is, but it can be adapted to work with commons OCR software such as Tesseract. The general idea is to search in the OCR'd text for a specific label using a fuzzy matching system and retrieve positional information (i.e. bounding boxes) for each word of the entity. The rest of the text is then labelles as other. 

`create_dataset.py`: creates the training, validation and testing splits according to the lists of documents provided by the authors.

`seg_bb_level.py`: creates an alternative version of the dataset for LayoutLMv3, which requires segments-level annotations.
