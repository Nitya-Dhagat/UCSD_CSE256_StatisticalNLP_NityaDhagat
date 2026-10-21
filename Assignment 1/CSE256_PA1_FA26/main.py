import argparse
import json
import os
import random
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader

from BOWmodels import SentimentDatasetBOW, NN2BOW, NN3BOW
from sentiment_data import read_word_embeddings, read_sentiment_examples
from DANmodels import (
    DAN, SentimentDatasetDAN, collate_dan, train_model, set_seed,
    build_train_indexer
)
from BPEmodels import (
    BPETokenizer, BPESentimentDataset, collate_bpe, SubwordDAN
)


# ----------------------------------------------------------------------------
# Generic helpers
# ----------------------------------------------------------------------------

def mean_std(values):
    arr = np.array(values, dtype=float)
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return float(arr.mean()), std


def summarize(name, results, target):
    """Aggregate a list of train_model() results (one per seed)."""
    best_mean, best_std = mean_std([r["best_dev_accuracy"] for r in results])
    reached = [r["epochs_to_target"] for r in results
               if r["epochs_to_target"] is not None]
    return {
        "name": name,
        "runs": len(results),
        "best_dev_mean": best_mean,
        "best_dev_std": best_std,
        "best_dev_max": max(r["best_dev_accuracy"] for r in results),
        "best_epoch_mean": mean_std([r["best_epoch"] for r in results])[0],
        "final_dev_mean": mean_std([r["final_dev_accuracy"] for r in results])[0],
        "train_acc_at_best_mean": mean_std([r["train_accuracy_at_best"] for r in results])[0],
        "final_train_mean": mean_std([r["final_train_accuracy"] for r in results])[0],
        "epochs_to_target_mean": (mean_std(reached)[0] if reached else None),
        "reached_target": len(reached),
        "min_dev_loss_epoch_mean": mean_std([r["min_dev_loss_epoch"] for r in results])[0],
        "train_time_mean": mean_std([r["train_time"] for r in results])[0],
        "time_to_best_mean": mean_std([r["time_to_best"] for r in results])[0],
        "params": results[0]["num_params"],
    }


def print_summaries(title, summaries, target):
    print("\n" + "=" * 150)
    print(title)
    print("(dev acc = best epoch selected on dev; 'ep->{:.0f}%' = first epoch with dev acc >= target, k/N = runs "
          "that reached it)".format(target * 100))
    print("=" * 150)
    header = "{:<44} {:>15} {:>7} {:>7} {:>8} {:>9} {:>8} {:>10} {:>8} {:>8} {:>9}".format(
        "Config", "best dev (m+-s)", "max", "bestEp", "lastDev", "lastTrain",
        "ep->tgt", "minLossEp", "time(s)", "t_best", "params")
    print(header)
    print("-" * len(header))
    for s in summaries:
        ep_tgt = "-"
        if s["epochs_to_target_mean"] is not None:
            ep_tgt = "{:.1f} ({}/{})".format(
                s["epochs_to_target_mean"], s["reached_target"], s["runs"])
        else:
            ep_tgt = "none (0/{})".format(s["runs"])
        print("{:<44} {:>15} {:>7.4f} {:>7.1f} {:>8.4f} {:>9.4f} {:>8} {:>10.1f} {:>8.1f} {:>8.1f} {:>9d}".format(
            s["name"][:44],
            "{:.4f}+-{:.4f}".format(s["best_dev_mean"], s["best_dev_std"]),
            s["best_dev_max"], s["best_epoch_mean"], s["final_dev_mean"],
            s["final_train_mean"], ep_tgt, s["min_dev_loss_epoch_mean"],
            s["train_time_mean"], s["time_to_best_mean"], s["params"]))


def mean_curve(results, key):
    return np.mean(np.array([r["history"][key] for r in results]), axis=0)


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)
    print("Saved", path)


# ----------------------------------------------------------------------------
# BOW
# ----------------------------------------------------------------------------

def train_epoch(data_loader, model, loss_fn, optimizer):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)

    model.train()
    train_loss, correct = 0, 0

    for batch, (X, y) in enumerate(data_loader):
        X = X.float()

        pred = model(X)
        loss = loss_fn(pred, y)

        train_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return correct / size, train_loss / num_batches


def eval_epoch(data_loader, model, loss_fn, optimizer):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)

    model.eval()
    eval_loss = 0
    correct = 0

    with torch.no_grad():
        for batch, (X, y) in enumerate(data_loader):
            X = X.float()
            pred = model(X)
            loss = loss_fn(pred, y)
            eval_loss += loss.item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    return correct / size, eval_loss / num_batches


def experiment(model, train_loader, test_loader):
    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    all_train_accuracy = []
    all_test_accuracy = []

    start = time.time()
    for epoch in range(100):
        train_accuracy, train_loss = train_epoch(
            train_loader, model, loss_fn, optimizer
        )
        all_train_accuracy.append(train_accuracy)

        test_accuracy, test_loss = eval_epoch(
            test_loader, model, loss_fn, optimizer
        )
        all_test_accuracy.append(test_accuracy)

        if epoch % 10 == 9:
            print(
                "Epoch #{}: train accuracy {:.3f}, dev accuracy {:.3f}".format(
                    epoch + 1, train_accuracy, test_accuracy
                )
            )

    return all_train_accuracy, all_test_accuracy, time.time() - start


def run_bow(args):
    start_time = time.time()

    set_seed(0)
    train_data = SentimentDatasetBOW(args.train_file)
    dev_data = SentimentDatasetBOW(
        args.dev_file, vectorizer=train_data.vectorizer, train=False
    )

    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    dev_loader = DataLoader(dev_data, batch_size=16, shuffle=False)

    print("Data loaded in: {:.2f} seconds".format(time.time() - start_time))

    runs = {}
    for name, builder in [
        ("BOW 2-layer", lambda: NN2BOW(input_size=512, hidden_size=100)),
        ("BOW 3-layer", lambda: NN3BOW(input_size=512, hidden_size=100)),
    ]:
        set_seed(0)
        print("\n{}:".format(name))
        model = builder()
        train_acc, dev_acc, elapsed = experiment(model, train_loader, dev_loader)
        best = max(dev_acc)
        runs[name] = {
            "train_accuracy": train_acc,
            "dev_accuracy": dev_acc,
            "best_dev_accuracy": best,
            "best_epoch": dev_acc.index(best) + 1,
            "final_train_accuracy": train_acc[-1],
            "final_dev_accuracy": dev_acc[-1],
            "train_time": elapsed,
            "num_params": sum(p.numel() for p in model.parameters()),
        }

    print("\n" + "=" * 100)
    print("BOW SUMMARY (100 epochs, Adam lr 1e-4, hidden 100, batch 16)")
    print("=" * 100)
    for name, r in runs.items():
        print("{:<14} best dev {:.4f} @ epoch {:>3} | last dev {:.4f} | final train {:.4f} | time {:.1f}s | params {}".format(
            name, r["best_dev_accuracy"], r["best_epoch"], r["final_dev_accuracy"],
            r["final_train_accuracy"], r["train_time"], r["num_params"]))

    plt.figure(figsize=(8, 6))
    plt.plot(runs["BOW 2-layer"]["train_accuracy"], label="2 layers")
    plt.plot(runs["BOW 3-layer"]["train_accuracy"], label="3 layers")
    plt.xlabel("Epochs")
    plt.ylabel("Training Accuracy")
    plt.title("Training Accuracy for 2 and 3 Layer Networks")
    plt.legend()
    plt.grid()
    plt.savefig("train_accuracy.png")
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(runs["BOW 2-layer"]["dev_accuracy"], label="2 layers")
    plt.plot(runs["BOW 3-layer"]["dev_accuracy"], label="3 layers")
    plt.xlabel("Epochs")
    plt.ylabel("Dev Accuracy")
    plt.title("Dev Accuracy for 2 and 3 Layer Networks")
    plt.legend()
    plt.grid()
    plt.savefig("dev_accuracy.png")
    plt.close()

    print("Saved train_accuracy.png and dev_accuracy.png")
    save_json(runs, "bow_results.json")


# ----------------------------------------------------------------------------
# DAN
# ----------------------------------------------------------------------------

def make_word_setup(indexer, embeddings, embedding_dim, args):
    train_data = SentimentDatasetDAN(args.train_file, indexer)
    dev_data = SentimentDatasetDAN(args.dev_file, indexer)
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_dan
    )
    dev_loader = DataLoader(
        dev_data, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_dan
    )
    return {
        "indexer": indexer, "embeddings": embeddings,
        "embedding_dim": embedding_dim,
        "train_data": train_data, "dev_data": dev_data,
        "train_loader": train_loader, "dev_loader": dev_loader,
    }


def run_dan_config(name, seeds, epochs, setup, args, use_pretrained,
                   model_kwargs=None, train_kwargs=None):
    model_kwargs = model_kwargs or {}
    train_kwargs = dict(train_kwargs or {})
    train_kwargs.setdefault("lr", args.lr)
    train_kwargs.setdefault("weight_decay", args.weight_decay)

    results = []
    for k, seed in enumerate(seeds):
        set_seed(seed)
        model = DAN(
            vocab_size=len(setup["indexer"]),
            embedding_dim=setup["embedding_dim"],
            word_embeddings=setup["embeddings"] if use_pretrained else None,
            **model_kwargs
        )
        print("\n--- {} | seed {} ---".format(name, seed))
        r = train_model(
            model, setup["train_loader"], setup["dev_loader"],
            epochs=epochs, target_dev_accuracy=args.target_acc,
            verbose=(k == 0), **train_kwargs
        )
        r["seed"] = seed
        results.append(r)
        print("seed {}: best dev {:.4f} @ epoch {} | last dev {:.4f} | train time {:.1f}s".format(
            seed, r["best_dev_accuracy"], r["best_epoch"],
            r["final_dev_accuracy"], r["train_time"]))
    return results


def run_dan_ablation(args, seeds, setups):
    def get_setup(key):
        if key in setups:
            return setups[key]
        path = args.glove_300
        if not os.path.exists(path):
            print("Skipping 300d configs: {} not found".format(path))
            return None
        emb = read_word_embeddings(path)
        setups[key] = make_word_setup(emb.word_indexer, emb, emb.get_embedding_length(), args)
        return setups[key]

    configs = [
        ("Baseline (50d, 1x100, drop .3, relu, adam 1e-3)", "50", {}, {}),
        ("GloVe 300d", "300", {}, {}),
        ("GloVe 50d frozen", "50", {"freeze_embeddings": True}, {}),
        ("Dropout 0.0", "50", {"dropout": 0.0}, {}),
        ("Dropout 0.5", "50", {"dropout": 0.5}, {}),
        ("Embedding(avg) dropout 0.2 + hidden .3", "50", {"embedding_dropout": 0.2}, {}),
        ("2 hidden layers", "50", {"num_hidden_layers": 2}, {}),
        ("Hidden size 300", "50", {"hidden_size": 300}, {}),
        ("Tanh activation", "50", {"activation": "tanh"}, {}),
        ("Adam lr 3e-4", "50", {}, {"lr": 3e-4}),
        ("Adam lr 3e-3", "50", {}, {"lr": 3e-3}),
        ("SGD(momentum .9) lr 0.05", "50", {}, {"optimizer_name": "sgd", "lr": 0.05}),
    ]

    summaries = []
    all_runs = {}
    for name, key, model_kwargs, train_kwargs in configs:
        setup = get_setup(key)
        if setup is None:
            continue
        results = run_dan_config(
            name, seeds, args.ablation_epochs, setup, args,
            use_pretrained=True, model_kwargs=model_kwargs, train_kwargs=train_kwargs
        )
        summaries.append(summarize(name, results, args.target_acc))
        all_runs[name] = results

    print_summaries(
        "DAN ABLATION ({} epochs, {} seeds)".format(args.ablation_epochs, len(seeds)),
        summaries, args.target_acc
    )
    save_json({"summaries": summaries, "runs": all_runs}, "dan_ablation_results.json")


def run_dan(args):
    start_time = time.time()
    seeds = list(range(args.seeds))

    glove50 = read_word_embeddings(args.glove_50)
    setups = {}
    setups["50"] = make_word_setup(
        glove50.word_indexer, glove50, glove50.get_embedding_length(), args
    )
    train_vocab_indexer = build_train_indexer(args.train_file)
    setups["train_vocab"] = make_word_setup(
        train_vocab_indexer, None, glove50.get_embedding_length(), args
    )

    s50 = setups["50"]
    print("\nWord-level data statistics")
    print("  GloVe/full vocabulary size (incl. PAD/UNK): {}".format(len(s50["indexer"])))
    print("  Train-only vocabulary size (incl. PAD/UNK): {}".format(len(train_vocab_indexer)))
    print("  Avg words per train sentence: {:.2f}".format(s50["train_data"].avg_length))
    print("  Dev UNK rate with GloVe vocab: {:.4f}".format(s50["dev_data"].unk_rate))
    print("  Dev UNK rate with train-only vocab: {:.4f}".format(setups["train_vocab"]["dev_data"].unk_rate))
    print("Training config: Adam lr={} weight_decay={} batch={} epochs={} hidden=100 dropout=0.3 seeds={}".format(
        args.lr, args.weight_decay, args.batch_size, args.epochs, seeds))

    glove_results = run_dan_config(
        "DAN GloVe-50d (fine-tuned)", seeds, args.epochs, s50, args, use_pretrained=True
    )
    random_results = run_dan_config(
        "DAN random-50d (full vocab)", seeds, args.epochs, s50, args, use_pretrained=False
    )
    random_tv_results = run_dan_config(
        "DAN random-50d (train vocab, OOV->UNK)", seeds, args.epochs,
        setups["train_vocab"], args, use_pretrained=False
    )

    summaries = [
        summarize("DAN GloVe-50d (fine-tuned)", glove_results, args.target_acc),
        summarize("DAN random-50d (full vocab)", random_results, args.target_acc),
        summarize("DAN random-50d (train vocab, OOV->UNK)", random_tv_results, args.target_acc),
    ]
    print_summaries(
        "DAN MAIN COMPARISON ({} epochs, {} seeds)".format(args.epochs, len(seeds)),
        summaries, args.target_acc
    )

    g, r = summaries[0], summaries[1]
    print("\nGloVe - random(full) gap in best dev acc: {:+.4f} (GloVe std {:.4f}, random std {:.4f})".format(
        g["best_dev_mean"] - r["best_dev_mean"], g["best_dev_std"], r["best_dev_std"]))
    print("Meets >= 77% requirement (mean best dev acc, GloVe): {}".format(
        "YES" if g["best_dev_mean"] >= 0.77 else "NO"))
    print("Total time for main comparison: {:.1f}s".format(time.time() - start_time))

    # Plots (mean over seeds)
    plt.figure(figsize=(8, 6))
    plt.plot(mean_curve(glove_results, "dev_accuracy"), label="GloVe")
    plt.plot(mean_curve(random_results, "dev_accuracy"), label="Random (full vocab)")
    plt.plot(mean_curve(random_tv_results, "dev_accuracy"), label="Random (train vocab)")
    plt.xlabel("Epoch")
    plt.ylabel("Dev Accuracy (mean over seeds)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("dan_dev_accuracy.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(mean_curve(glove_results, "train_loss"), label="GloVe train")
    plt.plot(mean_curve(glove_results, "dev_loss"), "--", label="GloVe dev")
    plt.plot(mean_curve(random_results, "train_loss"), label="Random train")
    plt.plot(mean_curve(random_results, "dev_loss"), "--", label="Random dev")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (mean over seeds)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("dan_loss.png", dpi=300)
    plt.close()
    print("Saved dan_dev_accuracy.png and dan_loss.png")

    save_json(
        {
            "summaries": summaries,
            "runs": {
                "glove50": glove_results,
                "random50_full_vocab": random_results,
                "random50_train_vocab": random_tv_results,
            },
        },
        "dan_results.json"
    )

    if args.ablation:
        run_dan_ablation(args, seeds, setups)


# ----------------------------------------------------------------------------
# Subword DAN (BPE)
# ----------------------------------------------------------------------------

EXAMPLE_WORDS = [
    "unbelievable", "disappointing", "masterpiece", "charming", "unfunny",
    "filmmaking", "overwrought", "wonderfully",
]


def bpe_diagnostics(tokenizer, args, suffix):
    """Print/save merge examples, real word splits, and why actual vocab < target."""
    lines = []

    def emit(text=""):
        print(text)
        lines.append(text)

    train_words = sorted({
        w for ex in read_sentiment_examples(args.train_file) for w in ex.words if w
    })
    train_set = set(train_words)
    dev_oov = sorted({
        w for ex in read_sentiment_examples(args.dev_file) for w in ex.words
        if w and w not in train_set
    })
    rng = random.Random(0)
    sampled_oov = rng.sample(dev_oov, min(8, len(dev_oov)))
    example_words = EXAMPLE_WORDS + sampled_oov

    emit("=" * 80)
    emit("BPE DIAGNOSTICS (keep_eow={})".format(tokenizer.keep_eow))
    emit("=" * 80)
    emit("Unique training word types: {}".format(tokenizer.num_training_words))
    emit("Initial symbols (chars + char</w> variants): {}".format(tokenizer.base_vocab_size))
    emit("Merge rules learned: {}".format(len(tokenizer.merges)))
    emit("Dev word types not in train: {}".format(len(dev_oov)))
    emit("")
    emit("First 25 merges: " + " | ".join("{}+{}".format(a, b) for a, b in tokenizer.merges[:25]))

    for v in args.vocab_sizes:
        applied = tokenizer.merges_applied(v)
        emit("")
        emit("--- target vocab {} ---".format(v))
        emit("Merges applied: {}".format(applied))
        last = tokenizer.merges[max(0, applied - 6):applied]
        emit("Last merges applied: " + " | ".join("{}+{}".format(a, b) for a, b in last))

        pieces_marked = set()
        pieces_stripped = set()
        for w in train_words:
            pieces_marked.update(tokenizer.tokenize_word(w, v, keep_eow=True))
            pieces_stripped.update(tokenizer.tokenize_word(w, v, keep_eow=False))
        emit("Distinct pieces in train WITH </w> kept: {}".format(len(pieces_marked)))
        emit("Distinct pieces in train after stripping </w>: {}".format(len(pieces_stripped)))
        emit("Target {} = base {} + merges {}; absorbed/unused intermediate symbols account for "
             "the gap to {} (+2 for PAD/UNK = {})".format(
                 v, tokenizer.base_vocab_size, applied, len(pieces_marked),
                 len(pieces_marked) + 2))
        emit("Embedding vocab for this run (keep_eow={}): {}".format(
            tokenizer.keep_eow,
            (len(pieces_marked) if tokenizer.keep_eow else len(pieces_stripped)) + 2))
        emit("Example splits:")
        for w in example_words:
            tag = "" if w in train_set else "  (not in train)"
            emit("  {:<18} -> {}{}".format(w, " | ".join(tokenizer.tokenize_word(w, v)), tag))

    path = "bpe_diagnostics{}.txt".format(suffix)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("Saved", path)


def run_subword_dan(args):
    start_time = time.time()
    seeds = list(range(args.seeds))
    suffix = "_keepeow" if args.keep_eow else ""

    print("\nSubword DAN: Byte Pair Encoding (keep_eow={})".format(args.keep_eow))
    print("Vocabulary sizes to test: {}".format(args.vocab_sizes))
    print("Training config: Adam lr={} weight_decay={} batch={} epochs={} emb={} hidden={} dropout={} seeds={}".format(
        args.lr, args.weight_decay, args.batch_size, args.epochs, args.embedding_dim,
        args.hidden_size, args.dropout, seeds))

    tokenizer = BPETokenizer(keep_eow=args.keep_eow)
    tokenizer.fit(args.train_file, vocab_size=max(args.vocab_sizes))
    bpe_diagnostics(tokenizer, args, suffix)

    summaries = []
    extras = []
    all_runs = {}
    mean_dev_curves = {}
    mean_train_curves = {}

    for vocab_size in args.vocab_sizes:
        print("\n" + "=" * 60)
        print("SUBWORD DAN with target BPE vocabulary size:", vocab_size)
        print("=" * 60)

        train_data = BPESentimentDataset(args.train_file, tokenizer, vocab_size)
        dev_data = BPESentimentDataset(
            args.dev_file, tokenizer, vocab_size, token_to_id=train_data.token_to_id
        )
        train_loader = DataLoader(
            train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_bpe
        )
        dev_loader = DataLoader(
            dev_data, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_bpe
        )

        actual_vocab = len(train_data.token_to_id)
        avg_tokens = train_data.average_tokens_per_sentence()
        print("Actual embedding vocabulary size:", actual_vocab)
        print("Avg subword tokens per train sentence: {:.2f}".format(avg_tokens))
        print("Dev UNK rate (subword tokens): {:.5f}".format(dev_data.unk_rate))

        results = []
        for k, seed in enumerate(seeds):
            set_seed(seed)
            model = SubwordDAN(
                vocab_size=actual_vocab,
                embedding_dim=args.embedding_dim,
                hidden_size=args.hidden_size,
                dropout=args.dropout
            )
            print("\n--- BPE {} | seed {} ---".format(vocab_size, seed))
            r = train_model(
                model, train_loader, dev_loader, epochs=args.epochs,
                lr=args.lr, weight_decay=args.weight_decay,
                target_dev_accuracy=args.target_acc, verbose=(k == 0)
            )
            r["seed"] = seed
            results.append(r)
            print("seed {}: best dev {:.4f} @ epoch {} | last dev {:.4f} | train time {:.1f}s".format(
                seed, r["best_dev_accuracy"], r["best_epoch"],
                r["final_dev_accuracy"], r["train_time"]))

        name = "SubwordDAN BPE {}{}".format(vocab_size, " (keep </w>)" if args.keep_eow else "")
        summaries.append(summarize(name, results, args.target_acc))
        extras.append({
            "target_vocab": vocab_size,
            "merges_applied": tokenizer.merges_applied(vocab_size),
            "actual_vocab": actual_vocab,
            "avg_tokens_per_sentence": avg_tokens,
            "dev_unk_rate": dev_data.unk_rate,
        })
        all_runs[name] = results

        dev_curve = mean_curve(results, "dev_accuracy")
        train_curve = mean_curve(results, "train_accuracy")
        mean_dev_curves[vocab_size] = dev_curve
        mean_train_curves[vocab_size] = train_curve

        plt.figure(figsize=(8, 6))
        plt.plot(dev_curve, label="Dev accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (mean over seeds)")
        plt.title("SubwordDAN Dev Accuracy, BPE {}".format(vocab_size))
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig("bpe_{}{}_dev_accuracy.png".format(vocab_size, suffix), dpi=300)
        plt.close()

        plt.figure(figsize=(8, 6))
        plt.plot(train_curve, label="Train accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (mean over seeds)")
        plt.title("SubwordDAN Train Accuracy, BPE {}".format(vocab_size))
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig("bpe_{}{}_train_accuracy.png".format(vocab_size, suffix), dpi=300)
        plt.close()

    # Combined plots across vocabulary sizes
    for curves, ylabel, fname in [
        (mean_dev_curves, "Dev accuracy (mean over seeds)", "bpe_all_dev_accuracy{}.png"),
        (mean_train_curves, "Train accuracy (mean over seeds)", "bpe_all_train_accuracy{}.png"),
    ]:
        plt.figure(figsize=(8, 6))
        for v, curve in curves.items():
            plt.plot(curve, label="BPE {}".format(v))
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(fname.format(suffix), dpi=300)
        plt.close()

    print_summaries(
        "SUBWORD DAN SUMMARY ({} epochs, {} seeds, keep_eow={})".format(
            args.epochs, len(seeds), args.keep_eow),
        summaries, args.target_acc
    )

    print("\nBPE vocabulary statistics")
    print("{:>8} {:>8} {:>12} {:>22} {:>12}".format(
        "target", "merges", "actualVocab", "avgTokens/sentence", "devUNKrate"))
    for e in extras:
        print("{:>8} {:>8} {:>12} {:>22.2f} {:>12.5f}".format(
            e["target_vocab"], e["merges_applied"], e["actual_vocab"],
            e["avg_tokens_per_sentence"], e["dev_unk_rate"]))

    print("\nTotal SUBWORDDAN experiment time: {:.2f} seconds".format(time.time() - start_time))
    save_json(
        {"summaries": summaries, "vocab_stats": extras, "runs": all_runs},
        "subword_results{}.json".format(suffix)
    )


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train BOW, DAN, or SUBWORDDAN sentiment models"
    )
    parser.add_argument("--model", type=str, required=True,
                        choices=["BOW", "DAN", "SUBWORDDAN"],
                        help="Model to train: BOW, DAN, or SUBWORDDAN")
    parser.add_argument("--vocab-sizes", type=int, nargs="+",
                        default=[1000, 2000, 3000, 5000, 8000],
                        help="BPE vocabulary sizes to experiment with")
    parser.add_argument("--keep-eow", action="store_true",
                        help="SUBWORDDAN: keep the </w> end-of-word marker in token ids")
    parser.add_argument("--seeds", type=int, default=3,
                        help="Number of random seeds (0..seeds-1) for DAN and SUBWORDDAN")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of epochs for DAN and SUBWORDDAN")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate (Adam) for DAN and SUBWORDDAN")
    parser.add_argument("--weight-decay", type=float, default=1e-5,
                        help="Weight decay for DAN and SUBWORDDAN")
    parser.add_argument("--target-acc", type=float, default=0.77,
                        help="Dev accuracy used for the 'epochs to target' statistic")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Training batch size for DAN and SUBWORDDAN")
    parser.add_argument("--eval-batch-size", type=int, default=128,
                        help="Development batch size for DAN and SUBWORDDAN")
    parser.add_argument("--embedding-dim", type=int, default=50,
                        help="Subword embedding dimension")
    parser.add_argument("--hidden-size", type=int, default=100,
                        help="Hidden layer size for SUBWORDDAN")
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="Dropout probability for SUBWORDDAN")
    parser.add_argument("--ablation", action="store_true",
                        help="DAN: also run the hyperparameter ablation grid")
    parser.add_argument("--ablation-epochs", type=int, default=10,
                        help="Epochs per run in the DAN ablation grid")
    parser.add_argument("--train-file", type=str, default="data/train.txt")
    parser.add_argument("--dev-file", type=str, default="data/dev.txt")
    parser.add_argument("--glove-50", type=str, default="data/glove.6B.50d-relativized.txt")
    parser.add_argument("--glove-300", type=str, default="data/glove.6B.300d-relativized.txt")

    args = parser.parse_args()

    if args.model == "BOW":
        run_bow(args)
    elif args.model == "DAN":
        run_dan(args)
    elif args.model == "SUBWORDDAN":
        run_subword_dan(args)


if __name__ == "__main__":
    main()