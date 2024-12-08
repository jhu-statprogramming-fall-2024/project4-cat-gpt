import os
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

input_file = './reddit_all.csv'

# generator
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

# discriminator
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

# inseting data
def get_text_embedding(texts, tokenizer, model, max_length=128, device="cpu"):
    model = model.to(device)
    embeddings = []
    for text in tqdm(texts, desc="Encoding texts"):
        tokens = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        input_ids = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]          embeddings.append(cls_output.squeeze(0).cpu().numpy())
    return np.array(embeddings)

# train the model
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
        # discrimiter
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

        # generator
        fake_data = generator(z)
        fake_outputs = discriminator(fake_data)
        g_loss = criterion(fake_outputs, real_labels)

        optimizer_g.zero_grad()
        g_loss.backward()
        optimizer_g.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}, D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

    return generator

# fill out missing scores
def process_data(data, tokenizer, bert_model, device="cpu"):
    texts = data['selftext'].fillna("").tolist()
    text_embeddings = get_text_embedding(texts, tokenizer, bert_model, device=device)

    # finding out the given score range
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

    data = process_data(data, tokenizer, bert_model, device=device)

    # save to the raw data
    data.to_csv(input_file, index=False)
    print(f"Updated data saved to {input_file}")

if __name__ == "__main__":
    main()
