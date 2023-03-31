from huggingface_hub import notebook_login
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer
from transformers import Seq2SeqTrainingArguments
from datasets import Dataset
import pandas as pd
import gc
import torch

class PravopysnykTrainer(object):
    """
    Trains a Ukrainian GEC model with the parameters passed.
    """
    def __init__(self, source_file, target_file, model_name):
        self.source_file = source_file
        self.target_file = target_file
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_checkpoint = "facebook/mbart-large-50"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint, src_lang="uk_UA", tgt_lang="uk_UA")

    def setup(self):
        """
        Sets everything up.
        """
        # logging into HF
        notebook_login()
        # setting device on GPU if available, else CPU
        print('Using device:', self.device)
        #Additional Info when using cuda
        if self.device.type == 'cuda':
            print(torch.cuda.get_device_name(0))
            print('Memory Usage:')
            print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
            print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
    
    def read_parallel_into_df(self, source_file, target_file):
        # reading the input data
        with open(source_file, 'r') as f:
            source = [line[:-1] for line in f.readlines()]
        with open(target_file, 'r') as f:
            target = [line[:-1] for line in f.readlines()]
        all_sentences = [[source[i], target[i]] for i in range(len(source))]
        df = pd.DataFrame(all_sentences)
        df.columns = ['source', 'target']
        return df

    def preprocess_function(self, examples, max_length=128):
        inputs = examples["source"]
        targets = examples["target"]
        model_inputs = self.tokenizer(
            inputs, text_target=targets, max_length=max_length, truncation=True
        )
        return model_inputs

    def load_data(self, source_file, target_file):
        """
        Loads the dataset into the correct format
        """
        train_df = self.read_parallel_into_df(source_file, target_file)
        raw_dataset = Dataset.from_pandas(train_df) # converting pandas dataframe to Dataset class
        tokenized_datasets = raw_dataset.map(
            self.preprocess_function,
            batched=True,
        )
        return tokenized_datasets

    def train(self, tokenized_datasets):
        """
        Actually trains the model.
        """
        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_checkpoint)
        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=model)
        args = Seq2SeqTrainingArguments(
            self.model_name,
            evaluation_strategy="no",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=64,
            weight_decay=0.01,
            save_total_limit=3,
            num_train_epochs=3,
            predict_with_generate=True,
            fp16=True,
            push_to_hub=True
        )
        trainer = Seq2SeqTrainer(
            model,
            args,
            train_dataset=tokenized_datasets,
            data_collator=data_collator,
            tokenizer=self.tokenizer
        )
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        trainer.train()
        print("Done!")

    def main(self):
        """
        Driver function for the entire process.
        """
        self.setup() # setting the basics up
        tokenized_datasets = self.load_data(self.source_file, self.target_file) # loading the data
        self.train(tokenized_datasets) # training the model