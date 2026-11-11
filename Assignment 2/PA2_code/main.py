import argparse
import glob
import os
import shutil

import matplotlib
matplotlib.use("Agg")  # no display needed; utilities.py still saves the PNGs

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from tokenizer import SimpleTokenizer
from dataset import SpeechesClassificationDataset, LanguageModelingDataset
from transformer import (TransformerEncoder, FeedforwardClassifier, EncoderClassifier,
                         TransformerDecoderLM)
from utilities import Utilities


seed = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" Hyperparameters to use for training to roughly match
the numbers mentioned in the assignment description """
batch_size = 16  # Number of independent sequences  we will process in parallel
block_size = 32  # Maximum context length for predictions
learning_rate = 1e-3  # Learning rate for the optimizer
n_embd = 64  # Embedding dimension
n_head = 2  # Number of attention heads
n_layer = 4  # Number of transformer layers

eval_interval = 100  # How often to evaluate train and test perplexity during training
max_iters = 500  # 500 * 16 * 32 = 256,000 tokens
eval_iters = 200  # Number of iterations to evaluate perplexity on the test set

## classifier training hyperparameters. It is a simple 1 hidden layer feedforward network, with input
## size of 64, hidden size of 100 and output size of 3.
n_input = 64  # Input size for the classifier, should match the embedding size of the transformer
n_hidden = 100  # Hidden size for the classifier
n_output = 3  # Output size for the classifier, we have 3 classes
epochs_CLS = 15  # epochs for classifier training

# Feedforward hidden size inside the Transformer blocks.
n_ff_decoder = 100        # specified in the assignment (Sec. 2.1)
n_ff_encoder = 4 * n_embd  # not specified for the encoder; standard 4x. Change if you like.

# Resolve the data folder relative to this file (not the current working directory),
# so `python main.py` works no matter where you launch it from. Override with the
# environment variable PA2_DATA_DIR if your folder lives somewhere else.
_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("PA2_DATA_DIR", os.path.join(_HERE, "speechesdataset"))


def load_texts(directory):
    """
    This function loads all texts from the specified directory, ignoring any files with "test" in their name. The text is used for "training" the tokenizer. Since our tokenizer is simple, we don't need to do any training, but we still need to ignore the test data.
    """
    texts = []
    files = os.listdir(directory)
    for filename in files:
        if "test" in filename:  ## don't "read test files"
            continue
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
            texts.append(file.read())
    return texts


def collate_batch(batch):
    """ Collate a batch of data into a single tensor with padding."""
    data, labels = zip(*batch)  # Separate the data and labels
    # Pad sequences to the fixed length
    padded_sequences = pad_sequence(data, batch_first=True, padding_value=0)
    padded_sequences = padded_sequences[:, :block_size]  # Truncate if longer
    # Add padding if shorter
    padded_sequences = torch.nn.functional.pad(padded_sequences, (0, max(0, block_size - padded_sequences.shape[1])), "constant", 0)
    labels = torch.stack(labels)
    return padded_sequences, labels


def compute_classifier_accuracy(classifier, data_loader):
    """ Compute the accuracy of the classifier on the data in data_loader."""
    classifier.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for X, Y in data_loader:
            X, Y = X.to(device), Y.to(device)
            outputs = classifier(X)
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == Y).sum().item()
            total_samples += Y.size(0)
    accuracy = (100 * total_correct / total_samples)
    classifier.train()
    return accuracy


def compute_perplexity(decoderLMmodel, data_loader, eval_iters=100):
    """ Compute the perplexity of the decoderLMmodel on the data in data_loader.
    Make sure to use the cross entropy loss for the decoderLMmodel.
    (The starter version referenced an undefined `total_loss`; removed.)
    """
    decoderLMmodel.eval()
    losses = []
    with torch.no_grad():
        for X, Y in data_loader:
            X, Y = X.to(device), Y.to(device)
            loss = decoderLMmodel(X, Y)  # cross-entropy loss
            losses.append(loss.item())
            if len(losses) >= eval_iters:
                break
    mean_loss = torch.tensor(losses).mean()
    perplexity = torch.exp(mean_loss).item()
    decoderLMmodel.train()
    return perplexity


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def run_sanity_check(tokenizer, model, sentence, outdir):
    """Run utilities.sanity_check (CPU), then move the saved PNGs into outdir."""
    os.makedirs(outdir, exist_ok=True)
    model.eval().cpu()
    with torch.no_grad():
        Utilities(tokenizer, model).sanity_check(sentence, block_size)
    model.to(device).train()
    for f in glob.glob("attention_map_*.png"):
        shutil.move(f, os.path.join(outdir, f))
    print(f"Attention maps saved to {outdir}/")


# --------------------------------------------------------------------------- #
# Part 1: encoder + classifier, trained jointly
# --------------------------------------------------------------------------- #
def part1(tokenizer):
    print("\n===== Part 1: Encoder + classifier =====")
    torch.manual_seed(seed)
    train_ds = SpeechesClassificationDataset(tokenizer, f"{DATA_DIR}/train_CLS.tsv")
    test_ds = SpeechesClassificationDataset(tokenizer, f"{DATA_DIR}/test_CLS.tsv")
    train_loader = DataLoader(train_ds, batch_size=batch_size, collate_fn=collate_batch, shuffle=True)
    train_eval_loader = DataLoader(train_ds, batch_size=batch_size, collate_fn=collate_batch, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, collate_fn=collate_batch, shuffle=False)

    encoder = TransformerEncoder(tokenizer.vocab_size, block_size, n_embd, n_head, n_layer, n_ff_encoder)
    clf = FeedforwardClassifier(n_input, n_hidden, n_output)
    model = EncoderClassifier(encoder, clf).to(device)
    print(f"Encoder parameters:    {count_params(encoder):,}")
    print(f"Classifier parameters: {count_params(clf):,}")
    print(f"Total parameters:      {count_params(model):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    history = []
    for epoch in range(1, epochs_CLS + 1):
        model.train()
        total_loss, n = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)
            n += xb.size(0)
        train_acc = compute_classifier_accuracy(model, train_eval_loader)
        test_acc = compute_classifier_accuracy(model, test_loader)
        history.append((epoch, total_loss / n, train_acc, test_acc))
        print(f"epoch {epoch:2d} | train loss {total_loss / n:.4f} | "
              f"train acc {train_acc:6.2f}% | test acc {test_acc:6.2f}%")
    print(f"Final test accuracy: {history[-1][3]:.2f}%")

    s1 = "We must never give up on the promise of this country."
    s2 = "The level of world cooperation and condemnation of Iraq is unprecedented."
    for i, s in enumerate([s1, s2], 1):
        print(f"\nSanity check, sentence {i}: {s}")
        run_sanity_check(tokenizer, model.encoder, s, f"part1_attention/sentence{i}")
    return model


# --------------------------------------------------------------------------- #
# Part 2: decoder language model
# --------------------------------------------------------------------------- #
def part2(tokenizer):
    print("\n===== Part 2: Decoder language model =====")
    torch.manual_seed(seed)

    def read(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    train_ds = LanguageModelingDataset(tokenizer, read(f"{DATA_DIR}/train_LM.txt"), block_size)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    train_eval_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    test_loaders = {}
    for name in ["obama", "wbush", "hbush"]:
        ds = LanguageModelingDataset(tokenizer, read(f"{DATA_DIR}/test_LM_{name}.txt"), block_size)
        test_loaders[name] = DataLoader(ds, batch_size=batch_size, shuffle=False)

    model = TransformerDecoderLM(tokenizer.vocab_size, block_size, n_embd, n_head, n_layer, n_ff_decoder).to(device)
    print(f"Decoder parameters: {count_params(model):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    def report(it):
        # train perplexity is measured on the first eval_iters batches of the
        # training stream (shuffle=True loader would also be fine)
        tr = compute_perplexity(model, train_eval_loader, eval_iters)
        te = {k: compute_perplexity(model, v, eval_iters) for k, v in test_loaders.items()}
        print(f"iter {it:4d} | train ppl {tr:8.2f} | obama {te['obama']:8.2f} | "
              f"wbush {te['wbush']:8.2f} | hbush {te['hbush']:8.2f}")

    model.train()
    for i, (xb, yb) in enumerate(train_loader):
        if i >= max_iters:
            break
        xb, yb = xb.to(device), yb.to(device)
        loss = model(xb, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (i + 1) % eval_interval == 0 or (i + 1) == max_iters:
            report(i + 1)

    s1 = "We must never give up on the promise of this country."
    s2 = "The level of world cooperation and condemnation of Iraq is unprecedented."
    for i, s in enumerate([s1, s2], 1):
        print(f"\nSanity check, sentence {i}: {s}")
        run_sanity_check(tokenizer, model, s, f"part2_attention/sentence{i}")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("parts", nargs="*", choices=["part1", "part2"],
                        help="which part(s) to run; default: both")
    args = parser.parse_args()
    parts = args.parts or ["part1", "part2"]

    if not os.path.isdir(DATA_DIR):
        raise SystemExit(f"Data folder not found: {DATA_DIR}\n"
                         "Put the 'speechesdataset' folder next to main.py, or set PA2_DATA_DIR.")
    print("Loading data and creating tokenizer ...")
    texts = load_texts(DATA_DIR)
    tokenizer = SimpleTokenizer(' '.join(texts))  # create a tokenizer from the data
    print("Vocabulary size is", tokenizer.vocab_size)

    if "part1" in parts:
        part1(tokenizer)
    if "part2" in parts:
        part2(tokenizer)


if __name__ == "__main__":
    main()