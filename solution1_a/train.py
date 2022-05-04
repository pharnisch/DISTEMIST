from flair.datasets import ColumnCorpus
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.embeddings import TransformerWordEmbeddings, FlairEmbeddings, StackedEmbeddings, WordEmbeddings
from torch.optim.lr_scheduler import OneCycleLR
import flair
import torch

flair.set_seed(1)

task = "ner"
columns = {0: "text", 1: "ner"}
data_folder = "../flair_style_training_data/"
# TODO: for full retraining at the end, sample_missing_splits=False
corpus = ColumnCorpus(data_folder, columns, train_file="all.txt")  #.downsample(0.1)

print(f"The training corpus contains {len(corpus.train)} (pretty long) sample sentences.")
print(f"The validation corpus contains {len(corpus.dev)} (pretty long) sample sentences.")
print(f"The testing corpus contains {len(corpus.test)} (pretty long) sample sentences.")

hidden_size = 128
stacked_embeddings = StackedEmbeddings(embeddings=[WordEmbeddings("es"), FlairEmbeddings("es-forward"), FlairEmbeddings("es-backward")]) # TransformerWordEmbeddings()
#roberta_embeddings = TransformerWordEmbeddings("roberta-base-bne")
dictionary = flair.data.Dictionary(add_unk=False)
dictionary.add_item("O")
dictionary.add_item("ENFERMEDAD")
tagger = SequenceTagger(stacked_embeddings, dictionary, task)

trainer = ModelTrainer(tagger, corpus)
trainer.train(
    base_path=f"taggers/{task}-stacked",
    train_with_dev=False,
    max_epochs=30,
    learning_rate=0.001,
    mini_batch_size=32,
    weight_decay=0.,
    embeddings_storage_mode="none",
    scheduler=OneCycleLR,
    optimizer=torch.optim.AdamW,
)