
import json
import numpy as np
from bert4keras.backend import keras, set_gelu, K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from bert4keras.layers import ConditionalRandomField
from keras.layers import Dense
from keras.models import Model
from tqdm import tqdm
import pylcs
from keras.layers import *

maxlen = 200
batch_size = 16
p = 'D:/Ai/model/electra-small/'
config_path = p +'bert_config_tiny.json'
checkpoint_path = p + 'electra_small'
dict_path = p +'vocab.txt'


def load_data(filename):
    D = []
    with open(filename,encoding = 'utf-8') as f:
        for l in f:
            l = json.loads(l)
            arguments = []
            for event in l['event_list']:
                arguments.append(label2id[event['event_type']])  
            D.append((l['text'], arguments))
    return D

# 读取schema
with open('event_schema.json',encoding = 'utf-8') as f:
    id2label, label2id, n = {}, {}, 0
    for l in f:
        l = json.loads(l)
        id2label[n] = l['event_type']
        label2id[l['event_type']] = n
        n += 1
    num_labels = len(id2label) + 1

tokenizer = Tokenizer(dict_path, do_lower_case=True)

class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, arguments) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text, max_length=maxlen)
            labels = [0] * num_labels
            for argument in arguments:
                labels[argument] = 1
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append(labels)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


bert = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    model='electra',
    return_keras_model=False,

)  # 建立模型，加载权重

output = Lambda(lambda x: x[:, 0],
                name='CLS-token')(bert.model.output)
output = Dense(units=num_labels,
               activation='sigmoid',
               kernel_initializer=bert.initializer)(output)

model = Model(bert.model.input, output)              

model.compile(loss='binary_crossentropy',
              optimizer=Adam(1e-5),
              metrics=['accuracy'])

train_data = load_data('train.json')
test_data = load_data('dev.json')
print(test_data)

train_generator = data_generator(train_data, batch_size)
test_generator = data_generator(test_data, 1)


def evaluate(data):
    total, right = 0., 0.
    count = 0
    for x_true, y_true in data:
        print('!!!!')
        print(count)
        count += 1
        print(x_true)
        y_pred = model.predict(x_true)
        for j in y_pred:
            res = []
            for i in range(len(j)):
                if j[i] > 0.5:
                    res.append(i)
            print(res) 
        # y_true = y_true[:, 0]
        # total += len(y_true)
        # right += (y_true == y_pred).sum()
    return 1

class Evaluator(keras.callbacks.Callback):
    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        val_acc = evaluate(test_generator)
        model.save_weights('best_model.weights')
        # if val_acc > self.best_val_acc:
        #     self.best_val_acc = val_acc
        # test_acc = evaluate(test_generator)
        # print(u'val_acc: %.5f, best_val_acc: %.5f, test_acc: %.5f\n'
        #       % (val_acc, self.best_val_acc, test_acc))





evaluator = Evaluator()
model.summary()
model.fit_generator(
    train_generator.forfit(),
    steps_per_epoch=len(train_generator),
    epochs=100,
    callbacks=[evaluator]
)