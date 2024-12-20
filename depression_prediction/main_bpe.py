import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from tokenizers.processors import TemplateProcessing

class TextDataset(Dataset):
    def __init__(self, texts, labels=None, max_length=200, tokenizer=None):
        self.texts = texts
        self.labels = labels
        self.max_length = max_length
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer.encode(self.texts[idx])
        ids = encoding.ids[:self.max_length]
        ids = ids + [0] * (self.max_length - len(ids))
        
        if self.labels is not None:
            return torch.tensor(ids), torch.tensor(self.labels[idx])
        return torch.tensor(ids)

class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout_rate=0.3):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, 
                           num_layers=num_layers, dropout=dropout_rate, bidirectional=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        lstm_out, _ = self.lstm(embedded)
        lstm_out = self.dropout(lstm_out)
        out = self.fc(lstm_out[:, -1, :])
        return self.sigmoid(out)

train_df = pd.read_csv('./train.csv')
test_df = pd.read_csv('./test.csv')

train_df['text'] = train_df['title'] + ' ' + train_df['body'].fillna('')
test_df['text'] = test_df['title'] + ' ' + test_df['body'].fillna('')

def create_bpe_tokenizer(texts, vocab_size=30000):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[PAD]", "[UNK]"],
    )
    
    tokenizer.train_from_iterator(texts, trainer=trainer)
    
    tokenizer.post_processor = TemplateProcessing(
        single="$A",
        special_tokens=[("[PAD]", 0), ("[UNK]", 1)],
    )
    
    return tokenizer

tokenizer = create_bpe_tokenizer(train_df['text'].values)

train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_df['text'].values, train_df['label'].values, test_size=0.2, random_state=42
)

train_dataset = TextDataset(train_texts, train_labels, tokenizer=tokenizer)
val_dataset = TextDataset(val_texts, val_labels, tokenizer=tokenizer)
test_dataset = TextDataset(test_df['text'].values, tokenizer=tokenizer)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTM(
    vocab_size=tokenizer.get_vocab_size(),
    embedding_dim=200,
    hidden_dim=256,
    num_layers=2,
    dropout_rate=0.3
).to(device)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=2
)

num_epochs = 20
best_val_acc = 0

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_texts, batch_labels in train_loader:
        batch_texts, batch_labels = batch_texts.to(device), batch_labels.float().to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_texts).squeeze()
        loss = criterion(outputs, batch_labels)
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    
    model.eval()
    val_preds = []
    val_true = []
    with torch.no_grad():
        for batch_texts, batch_labels in val_loader:
            batch_texts = batch_texts.to(device)
            outputs = model(batch_texts).squeeze()
            predictions = (outputs > 0.5).cpu().numpy()
            val_preds.extend(predictions)
            val_true.extend(batch_labels.numpy())
    
    val_accuracy = accuracy_score(val_true, val_preds)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')
    
    scheduler.step(val_accuracy)

model.eval()
test_preds = []
with torch.no_grad():
    for batch_texts in test_loader:
        batch_texts = batch_texts.to(device)
        outputs = model(batch_texts).squeeze()
        predictions = (outputs > 0.5).cpu().numpy().astype(int)
        test_preds.extend(predictions)

test_df['label'] = test_preds
test_df[['id', 'label']].to_csv('1.csv', index=False)
