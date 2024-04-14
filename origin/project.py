import argparse
import json
import math
from pprint import pprint
from random import randint
import random
import os
import openai

from datasets import load_dataset
import torch
from transformers import (AutoTokenizer,
                          AutoModelForCausalLM,
                          LogitsProcessorList,
                          AutoModelForSeq2SeqLM)
from extended_watermark_processor import WatermarkLogitsProcessor, WatermarkDetector


def str2bool(v):
    """Util function for user friendly boolean flag args"""
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

    parser = argparse.ArgumentParser(description="A minimum working example of applying the watermark to any LLM that supports the huggingface 🤗 `generate` API")

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="D:\\DDA4210\\facebookopt-1.3b"
    )
    parser.add_argument(
        "--prompt_max_length",
        type=int,
        default=None,
        help="Truncation length for prompt, overrides model config's max length field.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=200,
        help="Maximmum number of new tokens to generate.",
    )
    parser.add_argument(
        "--generation_seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--use_sampling",
        type=str2bool,
        default=True,
        help="Whether to generate using multinomial sampling.",
    )
    parser.add_argument(
        "--sampling_temp",
        type=float,
        default=0.7,
        help="Sampling temperature to use when generating using multinomial sampling.",
    )
    parser.add_argument(
        "--n_beams",
        type=int,
        default=1,
        help="Number of beams to use for beam search. 1 is normal greedy decoding",
    )
    parser.add_argument(
        "--use_gpu",
        type=str2bool,
        default=True,
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.25,
        help="The fraction of the vocabulary to partition into the greenlist at each generation and verification step.",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=2.0,
        help="The amount/bias to add to each of the greenlist token logits before each token sampling step.",
    )
    parser.add_argument(
        "--normalizers",
        type=str,
        default="",
        help="Single or comma separated list of the preprocessors/normalizer names to use when performing watermark detection.",
    )
    parser.add_argument(
        "--ignore_repeated_ngrams",     # penalty for repeated words
        type=str2bool,
        default=False,
        help="Whether to use the detection method that only counts each unqiue bigram once as either a green or red hit.",
    )
    parser.add_argument(
        "--detection_z_threshold",
        type=float,
        default=4.0,
    )
    parser.add_argument(
        "--load_fp16",
        type=str2bool,
        default=False,
        help="Whether to run model in float16 precsion.",
    )
    args = parser.parse_args()
    return args

def load_model(args):
    """Load and return the model and tokenizer"""
    
    args.is_decoder_only_model = any([(model_type in args.model_name_or_path) for model_type in ["gpt","opt","bloom"]])

    if args.is_decoder_only_model:
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    else:
        raise ValueError(f"Unknown model type: {args.model_name_or_path}")

    if args.use_gpu:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
    else:
        device = "cpu"
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    return model, tokenizer, device

def generate(prompt, args, model=None, device=None, tokenizer=None):
    
    watermark_processor = WatermarkLogitsProcessor(vocab=list(tokenizer.get_vocab().values()),
                                                    gamma=args.gamma,
                                                    delta=args.delta,
                                                    generation_seed = args.generation_seed
                                                    )

    if args.prompt_max_length:
        pass
    elif hasattr(model.config,"max_position_embedding"):    #whether model.config has name "max_position_embedding"
        args.prompt_max_length = model.config.max_position_embeddings - args.max_new_tokens
    else:
        args.prompt_max_length = 2048-args.max_new_tokens

    tokd_input = tokenizer(prompt, return_tensors="pt", add_special_tokens=True, truncation=True, max_length=args.prompt_max_length).to(device)
    truncation_warning = True if tokd_input["input_ids"].shape[-1] == args.prompt_max_length else False
    redecoded_input = tokenizer.batch_decode(tokd_input["input_ids"], skip_special_tokens=True)[0]
    
    gen_kwargs = dict(**tokd_input, max_new_tokens=args.max_new_tokens)

    if args.use_sampling:
        gen_kwargs.update(dict(
            do_sample=True, 
            top_k=0,
            temperature=args.sampling_temp
        ))
    else:
        gen_kwargs.update(dict(
            num_beams=args.n_beams
        ))
    output_without_watermark = model.generate(**gen_kwargs)
    output_with_watermark = model.generate(**gen_kwargs, 
                                            logits_processor=LogitsProcessorList([watermark_processor])
                                            )
    if args.is_decoder_only_model:
        # need to isolate the newly generated tokens
        output_without_watermark = output_without_watermark[:,tokd_input["input_ids"].shape[-1]:]
        output_with_watermark = output_with_watermark[:,tokd_input["input_ids"].shape[-1]:]
        
    
    decoded_output_without_watermark = tokenizer.batch_decode(output_without_watermark, skip_special_tokens=True)[0]
    decoded_output_with_watermark = tokenizer.batch_decode(output_with_watermark, skip_special_tokens=True)[0]

    return (redecoded_input,
            int(truncation_warning),
            decoded_output_without_watermark, 
            decoded_output_with_watermark,
            args) 


def list_format_scores(score_dict, detection_threshold):
    """Format the detection metrics into a gradio dataframe input format"""
    lst_2d = []
    for k,v in score_dict.items():
        if k=='green_fraction': 
            lst_2d.append([k, f"{v:.1%}"])
        elif k=='confidence': 
            lst_2d.append([k, f"{v:.3%}"])
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

def detect(input_text, args, device=None, tokenizer=None):
    """Instantiate the WatermarkDetection object and call detect on
        the input text returning the scores and outcome of the test"""
    watermark_detector = WatermarkDetector(vocab=list(tokenizer.get_vocab().values()),
                                        gamma=args.gamma,
                                        device=device,
                                        tokenizer=tokenizer,
                                        z_threshold=args.detection_z_threshold,
                                        normalizers=args.normalizers,
                                        ignore_repeated_ngrams=args.ignore_repeated_ngrams)
    if len(input_text) > 1:
        score_dict = watermark_detector.detect(input_text)
        output = list_format_scores(score_dict, watermark_detector.z_threshold)
    else:
        output = [["Error","string too short to compute metrics"]]
        output += [["",""] for _ in range(6)]
    return output

def compute_ppl(output_text, args, model=None, device = None, tokenizer=None):
    with torch.no_grad():
        tokd_inputs = tokenizer.encode(output_text, return_tensors="pt", add_special_tokens=True, truncation=True, max_length=args.prompt_max_length).to(device)
        tokd_labels = tokd_inputs.clone().detach()

        outputs = model(input_ids=tokd_inputs, labels=tokd_labels)
        loss = outputs.loss  # avg CE loss all positions (except -100, TODO plz check that this is working correctly)
        ppl = torch.tensor(math.exp(loss))
    return ppl

def attack(output_text):
    openai.api_key = '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'
    os.environ["http_proxy"] = "http://localhost:7890"
    os.environ["https_proxy"] = "http://localhost:7890"

    gpt_messages=[]
    gpt_messages.append({'role': 'user', 'content': 'Rewrite the following paragraph: ' + output_text})
    completion = openai.ChatCompletion.create(model = 'gpt-3.5-turbo',
                                            messages = gpt_messages,
                                            temperature = 0.5 )
    return completion['choices'][0]['message']['content']

def main(args): 
    """Run a command line version of the generation and detection operations
        and optionally launch and serve the gradio demo"""
    # Initial arg processing and log
    args.normalizers = (args.normalizers.split(",") if args.normalizers else [])

    model, tokenizer, device = load_model(args)

    attack_model_name = "D:\DDA4210\gpt"
    gptmodel = AutoModelForSeq2SeqLM.from_pretrained(attack_model_name)
    gpttokenizer = AutoTokenizer.from_pretrained(attack_model_name)

    with open("origin\ldfa.json", "r", encoding='utf-8') as f:
        text_list = [json.loads(line) for line in f.read().strip().split("\n")]
    sample_idx = random.randint(1, len(text_list))  # choose one prompt
    input_text =  text_list[sample_idx]['title']
    args.default_prompt =input_text
    term_width = 80
    print(input_text)

    _, _, decoded_output_without_watermark, decoded_output_with_watermark, _ = generate(input_text, 
                                                                                        args, 
                                                                                        model=model, 
                                                                                        device=device, 
                                                                                        tokenizer=tokenizer)
    # without_watermark_detection_result = detect(decoded_output_without_watermark, 
    #                                             args, 
    #                                             device=device, 
    #                                             tokenizer=tokenizer)
    # with_watermark_detection_result = detect(decoded_output_with_watermark, 
    #                                         args, 
    #                                         device=device, 
    #                                         tokenizer=tokenizer)
    # ppl_without_watermark = compute_ppl(decoded_output_without_watermark, 
    #                                       args,
    #                                       model=gptmodel,
    #                                       device=device, 
    #                                       tokenizer=gpttokenizer)
    # print(decoded_output_with_watermark)
    ppl_with_watermark = compute_ppl(decoded_output_with_watermark,
                                    args,
                                    model=gptmodel,
                                    device=device, 
                                    tokenizer=gpttokenizer)
    rewrite_watermark_result = attack(decoded_output_with_watermark, )
    print("生成文本"+decoded_output_with_watermark)
    print(ppl_with_watermark)
    print("重写文本"+rewrite_watermark_result)
    # print("#"*term_width)
    # print("Output without watermark:")
    # print(decoded_output_without_watermark)
    # print("-"*term_width)
    # print(f"Detection result @ {args.detection_z_threshold}:")
    # pprint(without_watermark_detection_result)
    # print("-"*term_width)

    # print("#"*term_width)
    # print("Output with watermark:")
    # print(decoded_output_with_watermark)
    # print("-"*term_width)
    # print(f"Detection result @ {args.detection_z_threshold}:")
    # pprint(with_watermark_detection_result)
    # print("-"*term_width)

    return

if __name__ == "__main__":

    args = parse_args()

    main(args)