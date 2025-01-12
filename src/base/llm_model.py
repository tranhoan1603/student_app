import torch
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline

#define quantization
nf4_config = BitsAndBytesConfig(
    load_in_4bit=True, 
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.float16
)

def get_hf_llm(model_name='microsoft/phi-2', max_new_tokens=1024, **kwargs):
    
    #Load pretrained model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=nf4_config,
        low_cpu_mem_usage=True
    )
    
    #Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    #Load pipeline
    model_pipeline = pipeline(
        'text-generation',
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id
    )

    #create huggingface pipeline
    llm = HuggingFacePipeline(
        pipeline=model_pipeline,
        model_kwargs = kwargs
    )

    return llm
