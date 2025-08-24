import os
import argparse
from datetime import datetime

import pandas as pd
from datasets import Dataset
from transformers import (
    T5TokenizerFast, T5ForConditionalGeneration,
    DataCollatorForSeq2Seq, Trainer, TrainingArguments
)

DEFAULT_MODEL = os.environ.get("BASE_MODEL", "google/t5-v1_1-small")

def load_data(csv_path: str, direction: str) -> Dataset:
    df = pd.read_csv(csv_path)
    if "direction" not in df.columns:
        raise ValueError("CSV must contain 'direction' column. Collect data from the app first.")
    df = df[df["direction"] == direction]
    if df.empty:
        raise ValueError(f"No rows found for direction={direction}")
    df = df.dropna(subset=["source_text", "corrected_output"]).reset_index(drop=True)
    return Dataset.from_pandas(df[["source_text", "corrected_output"]])

def preprocess_fn(examples, tokenizer, prefix, max_in=256, max_out=128):
    inputs = [prefix + x for x in examples["source_text"]]
    model_inputs = tokenizer(inputs, max_length=max_in, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["corrected_output"], max_length=max_out, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def train_direction(csv, base_model, out_dir, direction, prefix, epochs, batch, lr, fp16):
    ds = load_data(csv, direction)
    tokenizer = T5TokenizerFast.from_pretrained(base_model)
    model = T5ForConditionalGeneration.from_pretrained(base_model)

    tokenized = ds.map(lambda ex: preprocess_fn(ex, tokenizer, prefix),
                       batched=True, remove_columns=ds.column_names)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    subdir = direction.replace(" ", "_").replace("→", "to")
    save_path = os.path.join(out_dir, f"t5-{subdir}-{timestamp}")

    training_args = TrainingArguments(
        output_dir=save_path,
        learning_rate=lr,
        per_device_train_batch_size=batch,
        num_train_epochs=epochs,
        weight_decay=0.01,
        fp16=fp16,
        logging_steps=50,
        save_strategy="epoch",
        report_to=[]
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"✅ Trained {direction}, saved to {save_path}")
    return save_path

def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    print("=== Training English → Liberian English ===")
    enlib_path = train_direction(
        args.csv,
        args.base_model,
        args.out_dir,
        "English → Liberian English",
        prefix="translate to Liberian English: ",
        epochs=args.epochs,
        batch=args.batch,
        lr=args.lr,
        fp16=args.fp16,
    )

    print("=== Training Liberian English → English ===")
    liben_path = train_direction(
        args.csv,
        args.base_model,
        args.out_dir,
        "Liberian English → English",
        prefix="translate to English: ",
        epochs=args.epochs,
        batch=args.batch,
        lr=args.lr,
        fp16=args.fp16,
    )

    # Optional convenience symlinks/dirs for autoload
    latest_en = os.path.join(args.out_dir, "latest_en2lib")
    latest_lib = os.path.join(args.out_dir, "latest_lib2en")
    try:
        if os.path.islink(latest_en) or os.path.exists(latest_en):
            pass
        os.makedirs(latest_en, exist_ok=True)
        os.makedirs(latest_lib, exist_ok=True)
    except Exception:
        pass

    print("All done! Models saved:")
    print("EN→LIB:", enlib_path)
    print("LIB→EN:", liben_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/training_data.csv")
    parser.add_argument("--out_dir", default="outputs")
    parser.add_argument("--base_model", default=DEFAULT_MODEL)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()
    main(args)
