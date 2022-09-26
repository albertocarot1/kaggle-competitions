# %% [code] {"execution":{"iopub.status.busy":"2022-09-26T10:22:51.824031Z","iopub.execute_input":"2022-09-26T10:22:51.825630Z","iopub.status.idle":"2022-09-26T10:22:51.831904Z","shell.execute_reply.started":"2022-09-26T10:22:51.825589Z","shell.execute_reply":"2022-09-26T10:22:51.830782Z"}}
import torch, datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_error, mean_absolute_error
import pandas as pd
import numpy as np
from datasets import Dataset


# %% [code] {"execution":{"iopub.status.busy":"2022-09-26T10:22:51.834211Z","iopub.execute_input":"2022-09-26T10:22:51.835300Z","iopub.status.idle":"2022-09-26T10:22:51.843456Z","shell.execute_reply.started":"2022-09-26T10:22:51.835259Z","shell.execute_reply":"2022-09-26T10:22:51.842499Z"}}
def preprocess(dataset):
    dataset['text'] = dataset['text'].apply(lambda z: z.replace('\n', ' '))
    dataset['text'] = dataset['text'].apply(lambda z: z.replace('_', ' '))
    dataset['text'] = dataset['text'].apply(lambda z: z.replace('\r', ' '))
    dataset['text'] = dataset['text'].apply(lambda z: z.replace('\t', ' '))
    dataset['text'] = dataset['text'].apply(lambda x: ' '.join(x.split()))
    return dataset


# %% [code] {"execution":{"iopub.status.busy":"2022-09-26T10:22:51.846580Z","iopub.execute_input":"2022-09-26T10:22:51.846882Z","iopub.status.idle":"2022-09-26T10:22:52.187651Z","shell.execute_reply.started":"2022-09-26T10:22:51.846853Z","shell.execute_reply":"2022-09-26T10:22:52.186649Z"}}
dataset = pd.read_csv('feedback-prize-english-language-learning/train.csv')
dataset.drop(['text_id'], axis=1, inplace=True)
dataset.rename({'full_text': 'text'}, axis=1, inplace=True)
dataset = preprocess(dataset)

# %% [code] {"execution":{"iopub.status.busy":"2022-09-26T10:22:52.200146Z","iopub.execute_input":"2022-09-26T10:22:52.201106Z","iopub.status.idle":"2022-09-26T10:22:52.211507Z","shell.execute_reply.started":"2022-09-26T10:22:52.201061Z","shell.execute_reply":"2022-09-26T10:22:52.210538Z"}}
model_name = 'microsoft/deberta-v3-small'
test_size = 0.2
max_length = 200


# %% [code] {"execution":{"iopub.status.busy":"2022-09-26T10:22:52.190775Z","iopub.execute_input":"2022-09-26T10:22:52.191165Z","iopub.status.idle":"2022-09-26T10:22:52.198896Z","shell.execute_reply.started":"2022-09-26T10:22:52.191128Z","shell.execute_reply":"2022-09-26T10:22:52.197962Z"}}
def prepare_dataset(dataset):
    ds = Dataset.from_pandas(dataset)
    tok_ds = ds.map(tok_func, batched=True)
    splitted_data_set = tok_ds.train_test_split(test_size=0.3)
    return splitted_data_set


def tok_func(x): return tokenizer(x["text"], truncation=True, padding=True, max_length=max_length)


def compute_metrics_for_regression(eval_pred):
    logits, labels = eval_pred
    labels = labels.reshape(-1, 1)

    mse = mean_squared_error(labels, logits)
    rmse = mean_squared_error(labels, logits, squared=False)
    mae = mean_absolute_error(labels, logits)
    r2 = r2_score(labels, logits)
    smape = 1 / len(labels) * np.sum(2 * np.abs(logits - labels) / (np.abs(labels) + np.abs(logits)) * 100)

    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2, "smape": smape}


# %% [code] {"execution":{"iopub.status.busy":"2022-09-26T10:22:52.213186Z","iopub.execute_input":"2022-09-26T10:22:52.213584Z","iopub.status.idle":"2022-09-26T10:22:52.227402Z","shell.execute_reply.started":"2022-09-26T10:22:52.213536Z","shell.execute_reply":"2022-09-26T10:22:52.226430Z"}}
d1 = dataset[['text', 'cohesion']]
d2 = dataset[['text', 'syntax']]
d3 = dataset[['text', 'vocabulary']]
d4 = dataset[['text', 'phraseology']]
d5 = dataset[['text', 'grammar']]
d6 = dataset[['text', 'conventions']]
for d in [d1, d2, d3, d4, d5, d6]:
    d.columns = ['text', 'label']

# %% [code] {"execution":{"iopub.status.busy":"2022-09-26T10:22:52.228477Z","iopub.execute_input":"2022-09-26T10:22:52.229295Z","iopub.status.idle":"2022-09-26T10:23:12.422106Z","shell.execute_reply.started":"2022-09-26T10:22:52.229253Z","shell.execute_reply":"2022-09-26T10:23:12.420961Z"}}
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name,
                                                           num_labels=1,
                                                           problem_type="regression").to("cpu")

import os

os.environ["WANDB_DISABLED"] = "true"
# Specifiy the arguments for the trainer
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,
    per_device_train_batch_size=6,
    per_device_eval_batch_size=6,
    weight_decay=0.01,
    learning_rate=2e-5,
    logging_dir='./logs',
    save_total_limit=10,
    load_best_model_at_end=True,
    metric_for_best_model='rmse',
    evaluation_strategy="epoch",
    save_strategy="epoch",
    gradient_accumulation_steps=6
)

# %% [code] {"execution":{"iopub.status.busy":"2022-09-26T10:23:12.423612Z","iopub.execute_input":"2022-09-26T10:23:12.424062Z","iopub.status.idle":"2022-09-26T10:23:12.561811Z","shell.execute_reply.started":"2022-09-26T10:23:12.424025Z","shell.execute_reply":"2022-09-26T10:23:12.560846Z"}}
test_set = pd.read_csv('feedback-prize-english-language-learning/test.csv')
test_set.rename({'full_text': 'text'}, axis=1, inplace=True)
test_set = preprocess(test_set)
t_ds = Dataset.from_pandas(test_set).map(tok_func, batched=True)

# %% [code] {"execution":{"iopub.status.busy":"2022-09-26T10:23:12.563146Z","iopub.execute_input":"2022-09-26T10:23:12.564105Z","iopub.status.idle":"2022-09-26T10:55:05.601663Z","shell.execute_reply.started":"2022-09-26T10:23:12.564066Z","shell.execute_reply":"2022-09-26T10:55:05.599420Z"}}
import gc
import torch

predictions = []
for data_ in [d1, d2, d3, d4, d5, d6]:
    splitted_data_set = prepare_dataset(data_)
    # Call the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=splitted_data_set['train'], eval_dataset=splitted_data_set['test'],
        compute_metrics=compute_metrics_for_regression,
    )

    trainer.train()
    predictions.append(trainer.predict(t_ds).predictions.astype(float))

    del splitted_data_set, trainer
    gc.collect()
    torch.cuda.empty_cache()

# %% [code] {"execution":{"iopub.status.busy":"2022-09-26T10:55:05.608612Z","iopub.execute_input":"2022-09-26T10:55:05.609347Z","iopub.status.idle":"2022-09-26T10:55:05.634306Z","shell.execute_reply.started":"2022-09-26T10:55:05.609301Z","shell.execute_reply":"2022-09-26T10:55:05.633081Z"}}
sub = pd.DataFrame(predictions[0])
i = 1
for m in predictions[1:]:
    sub[i] = m
    i += 1

# %% [code] {"execution":{"iopub.status.busy":"2022-09-26T10:55:05.637052Z","iopub.execute_input":"2022-09-26T10:55:05.638153Z","iopub.status.idle":"2022-09-26T10:55:05.662314Z","shell.execute_reply.started":"2022-09-26T10:55:05.638068Z","shell.execute_reply":"2022-09-26T10:55:05.661450Z"}}
sub

# %% [code] {"execution":{"iopub.status.busy":"2022-09-26T10:55:05.664879Z","iopub.execute_input":"2022-09-26T10:55:05.665789Z","iopub.status.idle":"2022-09-26T10:55:05.697147Z","shell.execute_reply.started":"2022-09-26T10:55:05.665753Z","shell.execute_reply":"2022-09-26T10:55:05.696296Z"}}
submission = pd.read_csv('feedback-prize-english-language-learning/sample_submission.csv')
submission.iloc[:, 1:] = sub

# %% [code] {"execution":{"iopub.status.busy":"2022-09-26T10:55:05.698600Z","iopub.execute_input":"2022-09-26T10:55:05.698966Z","iopub.status.idle":"2022-09-26T10:55:05.709893Z","shell.execute_reply.started":"2022-09-26T10:55:05.698931Z","shell.execute_reply":"2022-09-26T10:55:05.708703Z"}}
submission.to_csv('feedback-prize-english-language-learning/submission.csv', index=False)

# %% [code] {"execution":{"iopub.status.busy":"2022-09-26T10:55:05.711404Z","iopub.execute_input":"2022-09-26T10:55:05.712196Z","iopub.status.idle":"2022-09-26T10:55:05.728114Z","shell.execute_reply.started":"2022-09-26T10:55:05.712155Z","shell.execute_reply":"2022-09-26T10:55:05.726966Z"}}
submission