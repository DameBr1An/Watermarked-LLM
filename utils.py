import argparse
import json
import os
from watermarkbase import WatermarkDetector, WatermarkLogitsWarper
import torch
from transformers import (AutoTokenizer, # type: ignore
                          AutoModelForCausalLM,
                          LogitsProcessorList,
                          AutoModelForSeq2SeqLM)
import random
from nltk.corpus import wordnet # type: ignore


def str2bool(v):
    """change flag strings to booleans if necessary"""

    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    """Command line argument specification"""

    parser = argparse.ArgumentParser(description="An example of applying the watermark to LLM")

    parser.add_argument("--model_name_or_path", type=str, default="D:\\DDA4210\\facebookopt-1.3b")
    parser.add_argument("--prompt_max_length", type=int, default=None, help="maximum length for prompt")
    parser.add_argument("--max_new_tokens", type=int, default=200,help="Maximmum number of generated tokens")
    parser.add_argument("--min_new_tokens", type=int, default=200,help="Minimmum number of generated tokens")
    parser.add_argument("--generation_seed", type=int, default=42)
    parser.add_argument("--use_gpu", type=str2bool, default=True)
    parser.add_argument("--gamma", type=float, default=0.25, help="The fraction of the vocabulary to partition into the greenlist")
    parser.add_argument("--delta",mtype=float, default=2, help="The strength/bias to add to each of the greenlist token logits before ampling")
    parser.add_argument("--detection_z_threshold", type=float, default=4.0)
    parser.add_argument("--load_fp16",type=str2bool,default=False,help="Whether to run model in float16 precsion.",)
    args = parser.parse_args()
    return args

def load_model(args):
    """Load the model and tokenizer for generation and perplexity computing"""
    
    args.is_decoder_only_model = any([(model_type in args.generate_model) for model_type in ["gpt","opt","bloom"]])

    if args.is_decoder_only_model:
        model = AutoModelForCausalLM.from_pretrained(args.generate_model)
    else:
        raise ValueError(f"Unknown model type: {args.generate_model}")

    if args.use_gpu:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        print('gpu success')
    else:
        device = "cpu"
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.generate_model)

    pplmodel = AutoModelForSeq2SeqLM.from_pretrained(args.ppl_model)
    pplmodel = pplmodel.to(device)
    ppltokenizer = AutoTokenizer.from_pretrained(args.ppl_model)

    return model, tokenizer, device, pplmodel, ppltokenizer

def load_prompts():
    """randomly select a question as prompt"""

    with open("lfqa.json", "r", encoding='utf-8') as f:
        prompts_data = json.load(f)
    sample_idx = random.randint(0,len(prompts_data)-1)
    input_text = prompts_data[sample_idx]['title']
    best_score = max(prompts_data[sample_idx]["answers"]["score"])
    answer_index = prompts_data[sample_idx]["answers"]["score"].index(best_score)
    original_answer = prompts_data[sample_idx]["answers"]["text"][answer_index]

    return input_text

def list_format_scores(score_dict, detection_threshold):
    """Format the detection metrics into a gradio dataframe input format"""

    lst_2d = []
    for k,v in score_dict.items():
        if k=='green_fraction': 
            lst_2d.append([k, f"{v:.1%}"])
        elif k=='num_tokens_scored':
            lst_2d.append([k+'(T)',v])
        elif k=='confidence': 
            lst_2d.append([k, f"{v:.3%}"])
        elif k=='green_token_mask': 
            lst_2d.append([k, v])
        elif isinstance(v, float): 
            lst_2d.append([k, f"{v:.3g}"])
        elif isinstance(v, bool):
            lst_2d.append([k, ("Watermarked" if v else "Human/Unwatermarked")])
        else: 
            lst_2d.append([k, f"{v}"])
    if "confidence" in score_dict:
        lst_2d.insert(-2,["z-score Threshold", f"{detection_threshold}"])
    else:
        lst_2d.insert(-1,["z-score Threshold", f"{detection_threshold}"])
    return lst_2d

def compute_ppl(output_text, args, model=None, device = None, tokenizer=None):
    with torch.no_grad():
        tokd_inputs = tokenizer.encode(output_text, return_tensors="pt", add_special_tokens=True, truncation=True, max_length=args.prompt_max_length).to(device)
        tokd_labels = tokd_inputs.clone().detach()
        outputs = model(input_ids=tokd_inputs, labels=tokd_labels)
        loss = outputs.loss
        perplexity = loss
    return perplexity

def refine(processor, tokenizer, text):
    top_n_words = processor.get_top_n_perplexity_words(tokenizer, n=5)
    for word, _ in top_n_words:
        ind =text.index(word)
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name())
        text[ind] = random.choice(synonyms)
    return text

def generate(prompt, args, model=None, device=None, tokenizer=None, refine = False):
    """generation with watermark"""

    watermark_processor = WatermarkLogitsWarper(fraction=args.gamma,
                                                    strength=args.delta,
                                                    vocab_size=model.config.vocab_size,
                                                    watermark_key=args.generation_seed)

    if args.prompt_max_length:
        pass
    elif hasattr(model.config,"max_position_embedding"):    #whether model.config has name "max_position_embedding"
        args.prompt_max_length = model.config.max_position_embeddings - args.max_new_tokens
    else:
        args.prompt_max_length = 2048-args.max_new_tokens

    tokd_input = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.prompt_max_length).to(device)
    redecoded_input = tokenizer.batch_decode(tokd_input["input_ids"], skip_special_tokens=True)[0]

    gen_kwargs = dict(**tokd_input, 
                        max_new_tokens=args.max_new_tokens,
                        min_length = args.min_new_tokens,
                        do_sample=True, 
                        temperature=0.7)
    output_without_watermark = model.generate(**gen_kwargs)
    output_with_watermark = model.generate(**gen_kwargs, logits_processor=LogitsProcessorList([watermark_processor]))
    
    output_without_watermark = output_without_watermark[:,tokd_input["input_ids"].shape[-1]:]
    output_with_watermark = output_with_watermark[:,tokd_input["input_ids"].shape[-1]:]

    if refine:  # whether use perplexity comperation to refine the generated text
        output_with_watermark = refine(output_with_watermark, watermark_processor, tokenizer)

    decoded_output_without_watermark = tokenizer.batch_decode(output_without_watermark, skip_special_tokens=True)[0]
    decoded_output_with_watermark = tokenizer.batch_decode(output_with_watermark, skip_special_tokens=True)[0]

    return (decoded_output_without_watermark, decoded_output_with_watermark) 


def detect(input_text, args, device=None, model = None, tokenizer=None):
    """Detect on the generated text returning the scores and outcome of the test"""
    watermark_detector = WatermarkDetector(fraction=args.gamma,
                                    strength=args.delta,
                                    vocab_size=model.config.vocab_size,
                                    watermark_key=args.generation_seed)
    gen_tokens = tokenizer(input_text, add_special_tokens=False)["input_ids"]
    if len(input_text) > 1:
        score_dict = watermark_detector.detect(gen_tokens, z_threshold=args.detection_z_threshold)
        output = list_format_scores(score_dict, args.detection_z_threshold)
        output[4][1] = tokenizer.decode(output[4][1]).split()   # generated watermarked words list
    return output


def paraphrasing_attack(output_text):
    """Use gpt-3.5-turbo to paraphrasing as an attack"""

    os.environ["OPENAI_API_BASE"] = 'https://api.xiaoai.plus/v1'
    os.environ["OPENAI_API_KEY"] = 'sk-A9YETsIlKFwB10fx50D8A174Df6f427891DdA451A829B352'
    from openai import OpenAI
    client = OpenAI(
        api_key = os.environ.get("OPENAI_API_KEY"),
        base_url = os.environ["OPENAI_API_BASE"]
    )
    gpt_messages=[]
    gpt_messages.append({'role': 'user', 'content': 'Rewrite the following paragraph without intro: ' + output_text})
    completion = client.chat.completions.create(model = 'gpt-3.5-turbo',
                                            messages = gpt_messages,
                                            temperature = 0.5)
    return completion.choices[0].message.content


def substitution_attack(sentence):
    """Use wordnet to substitute synonyms as an attack"""

    def get_synonyms(word):
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name())
        return list(synonyms)

    words = sentence.split()
    new_sentences = []
    for word in words:
        synonyms = get_synonyms(word)
        if synonyms and random.random() < 1/3:
            new_word = random.choice(synonyms)
            new_sentences.append(new_word)
        else:
            new_sentences.append(word)
    return ' '.join(new_sentences)