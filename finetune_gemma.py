import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    default_data_collator,
    EarlyStoppingCallback       
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model






def main():

    print("CUDA verfügbar:", torch.cuda.is_available())
    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Keine GPU")



    ds = load_dataset("entfane/psychotherapy", split="train")
    
    def extract_pairs_with_meta(example):
        illness = example["illness"]
        stage = example["stage"]
        dialog = example["conversations"]
        pairs = []
        # Präfix with meta infos
        meta_info = f"[Illness: {illness}] [Stage: {stage}] "
        context = meta_info
        for turn in dialog:
            speaker = turn["from"]
            text = turn["value"].strip()
            if speaker == "therapist":
                if context:
                    pairs.append({"Context": context.strip(), "Response": text})
                context += " " + text
            else:
                context += " " + text
        return pairs
    
    # all pairs
    all_pairs = []
    for ex in ds:
        all_pairs.extend(extract_pairs_with_meta(ex))
    
    
    # Flat-Dataset erstellen und auf 2000 limitieren
    flat_ds = Dataset.from_list(all_pairs).shuffle(seed=42).select(range(500))
    
    import random
    
    # test the dataset
    
    print(f"number of training pairs: {len(flat_ds)}")
    print(random.choice(flat_ds))
    
    
    # 2) Tokenizer & Model laden
    model_id  = "google/gemma-3-4b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # EOS als Pad-Token setzen
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    def tokenize_fn(ex):
        prompts = [
            ctx + "\n\n" + resp
            for ctx, resp in zip(ex["Context"], ex["Response"])
        ]
        tok = tokenizer(
            prompts,
            truncation=True,
            padding="max_length",
            max_length=1024
        )
        tok["labels"] = [ids.copy() for ids in tok["input_ids"]]
        return tok
    
    tokenized = flat_ds.map(tokenize_fn, batched=True)
    
    
    # 3) Quant-Config mit CPU-Offload
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=True,  # FP32-Gewichte auf CPU
        offload_buffers=True                     # Puffer auf CPU
    )
    
    # 4) Basis-Modell laden
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    
    # 5) PEFT-Setup (k-Bit + LoRA)
    model = prepare_model_for_kbit_training(base_model)
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_cfg)
    model.train()
    
    # 4. Train/Test Split für Early Stopping
    split = tokenized.train_test_split(test_size=0.1, seed=42)
    train_ds = split["train"]
    eval_ds  = split["test"]
    
    
    
    # 6. TrainingArguments mit LR-Scheduler & Early Stopping
    from transformers import IntervalStrategy, TrainingArguments, Trainer, EarlyStoppingCallback
    
    training_args = TrainingArguments(
        output_dir="./finetuned-gemma",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-4,
        num_train_epochs=10,
    
        # Logging
	logging_dir="./logs",                                           
        logging_strategy=IntervalStrategy.STEPS,
        logging_steps=20,
    
        # Evaluation
        eval_strategy=IntervalStrategy.STEPS,
        eval_steps=50,
    
   	# Checkpoints
   	save_strategy=IntervalStrategy.STEPS,
   	save_steps=100,
    	save_total_limit=45,
    
        # Early stopping / best model
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
    
        # LR-Scheduler
        warmup_steps=100,
        lr_scheduler_type="linear",
    
        # Mixed precision
        fp16=True,
        dataloader_num_workers=4,         
          dataloader_prefetch_factor=2,
          dataloader_persistent_workers=True
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        #callbacks=[EarlyStoppingCallback(early_stopping_patience=4)]
    )
    
    
    
    # Prüfe, ob alle Parameter auf CUDA sind
    not_on_cuda = []
    for name, param in model.named_parameters():
        if param.device.type != "cuda":
            not_on_cuda.append((name, param.device))
            break
    if not_on_cuda:
        print("Parameter nicht auf CUDA:", not_on_cuda[:5])
    else:
        print("Alle Parameter auf CUDA!")
    
    trainer.train(resume_from_checkpoint=True)
    print("Finetuning komplett!")
    print(result)

    


if __name__ == "__main__":
    main()








