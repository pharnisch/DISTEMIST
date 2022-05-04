from flair.datasets import ColumnCorpus
from flair.models import SequenceTagger
from flair.data import Sentence

columns = {0: "text", 1: "ner"}
data_folder = "../flair_style_training_data/"
corpus = ColumnCorpus(data_folder, columns, train_file="all.txt") #.downsample(0.1)

tagger = SequenceTagger.load("taggers/ner-stacked/final-model.pt")
s = Sentence("Confirmando masa nodular, siendo el tumor adenomatoide de epidídimo la primera posibilidad diagnóstica.")
tagger(s)
print(s)

result = tagger.evaluate(corpus.test)
print(result)