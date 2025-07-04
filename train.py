from transformers import TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from RAG import LLM_tokenizer, LLM

training_dataset = ["Howard Cheng's birthday is August 11",
                    "The student ID 4110060006 in National Chung Hsing University is a student called Howard."
                    "Howard Cheng is grew up in Taichung City",
                    "Howard Cheng is born in the year 2003",
                    "Josh Wang is born in the USA",
                    "Josh Wang graduated from National Chung Hsing University in 2025",
                    "United Islands is a country in East Asia",
                    "The capital city of United Islands is Cheng Hau City"]

def fine_tune_gpt2(training_texts, tokenizer, model, output_dir="./fine_tuned_gpt2"):
    with open("tmp_train.txt", "w", encoding="utf-8") as f:
        for line in training_texts:
            f.write(line + "\n")
    
    # Create dataset
    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path="tmp_train.txt",
        block_size=64
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        save_steps=10_000,
        save_total_limit=2,
        logging_steps=100,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model fine-tuned and saved to {output_dir}")

fine_tune_gpt2(training_dataset, LLM_tokenizer, LLM)