"""Optional Qwen 0.5B LoRA adapter inference and training. Requires extra libraries/model download.
Explicit --mode is required; no downloads or GPU access occur merely by importing this module.
"""
from pathlib import Path
import argparse
ROOT=Path(__file__).resolve().parent
MODEL_ID='Qwen/Qwen2.5-0.5B-Instruct'

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--mode',choices=['train','infer'],required=True)
    ap.add_argument('--base-model',default=MODEL_ID)
    ap.add_argument('--adapter',type=Path,default=ROOT/'checkpoint')
    ap.add_argument('--text',default='I like to learn new things.')
    ap.add_argument('--train-limit',type=int,default=60)
    args=ap.parse_args()
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel, LoraConfig, get_peft_model, TaskType
    except ImportError as e:ap.error('Please install transformers, peft, accelerate, datasets, torch. '+str(e))
    tokenizer=AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token=tokenizer.eos_token
    model=AutoModelForCausalLM.from_pretrained(args.base_model)
    prompt=lambda sentence:f'### Instruction:\nRewrite this in the style of Master Yoda.\n\n### Input:\n{sentence}\n\n### Response:\n'
    if args.mode=='infer':
        if not (args.adapter/'adapter_config.json').is_file():ap.error('Local adapter checkpoint missing')
        model=PeftModel.from_pretrained(model,str(args.adapter));model.eval()
        inputs=tokenizer(prompt(args.text),return_tensors='pt')
        with torch.no_grad():out=model.generate(**inputs,max_new_tokens=60,do_sample=False,pad_token_id=tokenizer.eos_token_id)
        print(tokenizer.decode(out[0,inputs['input_ids'].shape[1]:],skip_special_tokens=True))
        return
    from datasets import load_dataset
    from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
    # The dataset is downloaded by this explicit training command; network and compute required.
    data=load_dataset('dvgodoy/yoda_sentences',split=f'train[:{args.train_limit}]')
    def encode(row):
        inp=tokenizer(prompt(row['sentence'])+row['translation']+tokenizer.eos_token,
                      truncation=True,max_length=256)
        # Causal-LM training loss masks pad automatically through collator
        return inp
    data=data.map(encode,remove_columns=data.column_names)
    model=get_peft_model(model,LoraConfig(r=8,lora_alpha=16,lora_dropout=.05,
        target_modules=['q_proj','v_proj'],task_type=TaskType.CAUSAL_LM))
    out=ROOT.parents[1]/'outputs'/'14_lora_training';out.mkdir(parents=True,exist_ok=True)
    # version compatibility: training API supported by transformers 4.57.x
    ta=TrainingArguments(output_dir=str(out),num_train_epochs=1,per_device_train_batch_size=1,
                         gradient_accumulation_steps=4,learning_rate=2e-4,logging_steps=10,
                         save_strategy='no',report_to='none')
    collator=DataCollatorForLanguageModeling(tokenizer=tokenizer,mlm=False)
    trainer=Trainer(model=model,args=ta,train_dataset=data,data_collator=collator)
    trainer.train();model.save_pretrained(out/'adapter');tokenizer.save_pretrained(out/'adapter')
    print('LoRA adapter saved:',out/'adapter')
if __name__=='__main__':main()
