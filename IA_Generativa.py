# Inteligência Artificial Generativa BERT
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import transformers
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from transformers import DistilBertTokenizer, DistilBertModel, AdamW
import torch.optim.lr_scheduler as lr_scheduler

# Habilitar o lançamento de bloqueio CUDA para depuração
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Verificar disponibilidade da GPU ou usar CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carregar dados e transformar as perguntas em rótulos numéricos
df = pd.read_csv('Conversation.csv')
df.drop(df.columns[0], axis=1, inplace=True)
le = LabelEncoder()
df['question'] = le.fit_transform(df['question'])
train_text, train_labels = df['answer'], df['question']

# Tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Definir o número máximo de tokens nas sequências
max_seq_len = 32

# Tokenizar os textos
tokens_train = tokenizer(
    train_text.tolist(),
    max_length=max_seq_len,
    padding='max_length',
    truncation=True,
    return_tensors='pt'
)
train_seq = tokens_train['input_ids']
train_mask = tokens_train['attention_mask']
train_y = torch.tensor(train_labels.tolist())

# DataLoader
batch_size = 1  # Reduza ainda mais o tamanho do lote se necessário
train_data = TensorDataset(train_seq, train_mask, train_y)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# Modelo BERT
class BERT_Arch(nn.Module):
    def __init__(self, bert):
        super(BERT_Arch, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 5)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sent_id, mask):
        cls_hs = self.bert(sent_id, attention_mask=mask)[0][:, 0]
        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x

# Carregar modelo BERT pré-treinado
bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
model = BERT_Arch(bert)
model = model.to(device)

# Otimizador e learning rate scheduler
optimizer = AdamW(model.parameters(), lr=1e-3)
lr_sch = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

# Calcular pesos de classe manualmente e aplicar na loss function
class_wts = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float)
class_wts = class_wts.to(device)

# Loss function
cross_entropy = nn.NLLLoss(weight=class_wts)

# Treinamento
train_losses = []
epochs = 200

def train():
    model.train()
    total_loss = 0
    total_preds = []

    for step, batch in enumerate(train_dataloader):
        if step % 50 == 0 and not step == 0:
            print('Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))

        batch = [r.to(device) for r in batch]
        sent_id, mask, labels = batch
        preds = model(sent_id, mask)

        loss = cross_entropy(preds, labels)

        total_loss = total_loss + loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        preds = preds.detach().cpu().numpy()
        total_preds.append(preds)

    avg_loss = total_loss / len(train_dataloader)
    total_preds = np.concatenate(total_preds, axis=0)
    return avg_loss, total_preds

for epoch in range(epochs):
    print('\nEpoch {:} / {:}'.format(epoch + 1, epochs))
    train_loss, _ = train()
    train_losses.append(train_loss)
    lr_sch.step()

print(f'\nTraining Loss: {train_loss:.3f}')
