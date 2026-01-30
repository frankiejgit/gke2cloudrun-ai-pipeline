import os
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, DistilBertForSequenceClassification
from datasets import load_dataset
from google.cloud import storage
import glob

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Environmental variables
MODEL_NAME = os.getenv("MODEL_NAME", "distilbert-base-uncased")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "2e-5"))
EPOCHS = int(os.getenv("EPOCHS", "3"))
BUCKET_NAME = os.getenv("BUCKET_NAME")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "model_output") # GCS path prefix

def upload_to_gcs(bucket_name, source_directory, destination_blob_prefix):
    """Uploads a directory to the Google Cloud Storage bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    for root, dirs, files in os.walk(source_directory):
        for filename in files:
            local_path = os.path.join(root, filename)
            relative_path = os.path.relpath(local_path, source_directory)
            blob_path = os.path.join(destination_blob_prefix, relative_path)

            blob = bucket.blob(blob_path)
            blob.upload_from_filename(local_path)
            logger.info(f"Uploaded {local_path} to gs://{bucket_name}/{blob_path}")

def main():
    # Initialize distributed training
    # Expects MASTER_ADDR, MASTER_PORT, RANK, WORLD_SIZE to be set in env
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        torch.distributed.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        logger.info("Not using distributed training (single GPU or CPU)")
        local_rank = 0
        rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Process rank: {rank}, Local rank: {local_rank}, World size: {world_size}")

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Load the SST-2 dataset
    # In a real scenario, you might want to download data to local disk first or stream it
    dataset = load_dataset("sst2")

    # Preprocess the data
    def preprocess_function(examples):
        return tokenizer(examples["sentence"], padding="max_length", truncation=True)

    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # Create data loaders with distributed sampler
    if world_size > 1:
        train_sampler = DistributedSampler(tokenized_dataset["train"], num_replicas=world_size, rank=rank)
    else:
        train_sampler = None

    train_loader = DataLoader(
        tokenized_dataset["train"], 
        sampler=train_sampler,
        batch_size=BATCH_SIZE, 
        shuffle=(train_sampler is None)
    )

    # Load model
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(device)

    if world_size > 1:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    # Optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(EPOCHS):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        
        model.train()
        for idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            # Logging
            if idx % 100 == 0 and rank == 0:
                logger.info(f"Epoch: {epoch+1}/{EPOCHS}, Batch: {idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

    # Save the trained model
    if rank == 0: # Only save on the main process
        logger.info("Training complete. Saving model...")
        local_output_dir = "./trained_model"
        
        if world_size > 1:
            model.module.save_pretrained(local_output_dir)
        else:
            model.save_pretrained(local_output_dir)
            
        tokenizer.save_pretrained(local_output_dir)

        if BUCKET_NAME:
            logger.info(f"Uploading model to gs://{BUCKET_NAME}/{OUTPUT_DIR}")
            upload_to_gcs(BUCKET_NAME, local_output_dir, OUTPUT_DIR)
        else:
            logger.warning("BUCKET_NAME not set. Model saved locally but not uploaded to GCS.")

if __name__ == "__main__":
    main()
