import os
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

input_file = "D:/pythonProject/reddit_all.csv"
output_file = "D:/pythonProject/reddit_all_processed.csv"  

# Generator 
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

# Discriminator 
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)

）
def get_text_embedding(texts, tokenizer, model, max_length=128, batch_size=1, device="cpu"):
    model = model.to(device)
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding texts"):
        batch_texts = texts[i:i+batch_size]
        tokens = tokenizer(
            batch_texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        input_ids = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  
        embeddings.extend(cls_output.cpu().numpy())
    return np.array(embeddings)

def train_gan(data, text_embeddings, score_min, score_max, epochs=50, lr=0.0002, device="cpu"):
    generator = Generator(input_dim=text_embeddings.shape[1], output_dim=1).to(device)
    discriminator = Discriminator(input_dim=1).to(device)

    criterion = nn.BCELoss()
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=lr)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr)

    real_scores = torch.tensor(data['sentiment_score'].dropna().values, dtype=torch.float32).unsqueeze(1).to(device)
    text_embeddings = torch.tensor(text_embeddings, dtype=torch.float32).to(device)
    batch_size = real_scores.size(0)

    for epoch in range(epochs):
        
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        outputs = discriminator(real_scores)
        d_loss_real = criterion(outputs, real_labels)

        z = text_embeddings[:batch_size]
        fake_data = generator(z)
        fake_outputs = discriminator(fake_data.detach())
        d_loss_fake = criterion(fake_outputs, fake_labels)

        d_loss = d_loss_real + d_loss_fake
        optimizer_d.zero_grad()
        d_loss.backward()
        optimizer_d.step()

        
        fake_data = generator(z)
        fake_outputs = discriminator(fake_data)
        g_loss = criterion(fake_outputs, real_labels)

        optimizer_g.zero_grad()
        g_loss.backward()
        optimizer_g.step()

        
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"D Real Scores: {outputs.detach().cpu().numpy().flatten()[:5]}")  # 显示前 5 个真实分数
        print(f"D Fake Scores: {fake_outputs.detach().cpu().numpy().flatten()[:5]}")  # 显示前 5 个虚假分数
        print(f"G Generated Scores: {fake_data.detach().cpu().numpy().flatten()[:5]}")  # 显示前 5 个生成分数
        print(f"D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

    return generator


def process_data(data, tokenizer, bert_model, device="cpu", max_length=128, batch_size=1):
    texts = data['selftext'].fillna("").tolist()
    text_embeddings = get_text_embedding(texts, tokenizer, bert_model, max_length=max_length, batch_size=batch_size, device=device)

    
    score_min = data['sentiment_score'].min()
    score_max = data['sentiment_score'].max()
    print(f"Detected score range: [{score_min}, {score_max}]")

    generator = train_gan(data, text_embeddings, score_min, score_max, device=device)

    unmarked_indices = data['sentiment_score'].isna()
    if unmarked_indices.any():
        missing_embeddings = text_embeddings[unmarked_indices]
        missing_embeddings = torch.tensor(missing_embeddings, dtype=torch.float32).to(device)
        fake_scores = generator(missing_embeddings).detach().cpu().numpy().flatten()
        data.loc[unmarked_indices, 'sentiment_score'] = np.clip(fake_scores, score_min, score_max)

    return data


def main():
    if not os.path.exists(input_file):
        print(f"File {input_file} not found. Please place the file in the correct location.")
        return

    data = pd.read_csv(input_file)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    bert_model = AutoModel.from_pretrained("bert-base-uncased")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = process_data(data, tokenizer, bert_model, device=device, max_length=128, batch_size=1)

    
    data.to_csv(output_file, index=False)
    print(f"Processed data saved to {output_file}")


if __name__ == "__main__":
    main()
