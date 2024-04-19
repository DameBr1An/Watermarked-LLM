import argparse
import json
import math
import os

from gptwm import WatermarkDetector, WatermarkLogitsWarper
import torch
from transformers import (AutoTokenizer,
                          AutoModelForCausalLM,
                          LogitsProcessorList,
                          AutoModelForSeq2SeqLM)
# from watermark_processor import WatermarkLogitsProcessor, WatermarkDetector

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
    
    # watermark_processor = WatermarkLogitsProcessor(vocab=list(tokenizer.get_vocab().values()),
    #                                                 gamma=args.gamma,
    #                                                 delta=args.delta,
    #                                                 generation_seed = args.generation_seed
    #                                                 )
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

    tokd_input = tokenizer(prompt, return_tensors="pt", add_special_tokens=True, truncation=True, max_length=args.prompt_max_length).to(device)
    truncation_warning = True if tokd_input["input_ids"].shape[-1] == args.prompt_max_length else False
    redecoded_input = tokenizer.batch_decode(tokd_input["input_ids"], skip_special_tokens=True)[0]
    
    # with open('greenlist.txt', 'w') as file:
    #     lines = file.write('')
    
    gen_kwargs = dict(**tokd_input, max_new_tokens=args.max_new_tokens)
    if args.use_sampling:
        gen_kwargs.update(dict(do_sample=True, top_k=0, temperature=args.sampling_temp))
    else:
        gen_kwargs.update(dict(num_beams=args.n_beams))

    output_without_watermark = model.generate(**gen_kwargs)
    output_with_watermark = model.generate(**gen_kwargs, logits_processor=LogitsProcessorList([watermark_processor]))
    # print(watermark_processor.list_of_greenlist_ids)
    if args.is_decoder_only_model:
        # need to isolate the newly generated tokens
        output_without_watermark = output_without_watermark[:,tokd_input["input_ids"].shape[-1]:]
        output_with_watermark = output_with_watermark[:,tokd_input["input_ids"].shape[-1]:]

    # with open('greenlist.txt', 'r') as file:
    #     total_green_list = file.readline().split()
    # output_list = output_with_watermark.tolist()[0]

    # if len(total_green_list) == len(output_list):
    #     tk_wm_list = []
    #     for i in range(len(output_list)):
    #         if output_list[i] in total_green_list[i]:
    #             tk_wm_list.append(output_list[i])
        # tk_wm_list = [int(item) for item in total_green_list if item != '-1']
        # word_list = tokenizer.decode(tk_wm_list).split()
        # print(word_list)

    decoded_output_without_watermark = tokenizer.batch_decode(output_without_watermark, skip_special_tokens=True)[0]
    decoded_output_with_watermark = tokenizer.batch_decode(output_with_watermark, skip_special_tokens=True)[0]

    return (decoded_output_without_watermark, decoded_output_with_watermark) 


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

def detect(input_text, args, device=None, model = None, tokenizer=None):
    """Instantiate the WatermarkDetection object and call detect on
        the input text returning the scores and outcome of the test"""
    # watermark_detector = WatermarkDetector(vocab=list(tokenizer.get_vocab().values()),
    #                                     gamma=args.gamma,
    #                                     device=device,
    #                                     tokenizer=tokenizer,
    #                                     z_threshold=args.detection_z_threshold,
    #                                     normalizers=args.normalizers,
    #                                     ignore_repeated_ngrams=args.ignore_repeated_ngrams)
    watermark_detector = WatermarkDetector(fraction=args.gamma,
                                    strength=args.delta,
                                    vocab_size=model.config.vocab_size,
                                    watermark_key=args.generation_seed)
    gen_tokens = tokenizer(input_text, add_special_tokens=False)["input_ids"]
    if len(input_text) > 1:
        score_dict = watermark_detector.detect(gen_tokens, z_threshold=args.detection_z_threshold)
        output = list_format_scores(score_dict, args.detection_z_threshold)
        output[4][1] = tokenizer.decode(output[4][1]).split()
    else:
        output = [["Error","string too short to compute metrics"]]
        output += [["",""] for _ in range(6)]
    return output

def compute_ppl(output_text, args, model=None, device = None, tokenizer=None):
    with torch.no_grad():
        tokd_inputs = tokenizer.encode(output_text, return_tensors="pt", add_special_tokens=True, truncation=True, max_length=args.prompt_max_length).to(device)
        tokd_labels = tokd_inputs.clone().detach()
        outputs = model(input_ids=tokd_inputs, labels=tokd_labels)
        loss = outputs.loss
        perplexity = math.exp(loss)
    return perplexity

def attack(output_text):
    # os.environ["OPENAI_API_KEY"] = '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'  #官网
    os.environ["OPENAI_API_BASE"] = 'https://api.xiaoai.plus/v1'
    os.environ["OPENAI_API_KEY"] = 'sk-A9YETsIlKFwB10fx50D8A174Df6f427891DdA451A829B352'
    from openai import OpenAI
    # openai.api_key = 'sk-A9YETsIlKFwB10fx50D8A174Df6f427891DdA451A829B352'
    # os.environ["http_proxy"] = "http://localhost:7890"
    # os.environ["https_proxy"] = "http://localhost:7890"
    client = OpenAI(
        # This is the default and can be omitted
        api_key = os.environ.get("OPENAI_API_KEY"),
        base_url = os.environ["OPENAI_API_BASE"]
    )
    gpt_messages=[]
    gpt_messages.append({'role': 'user', 'content': 'Rewrite the following paragraph without intro: ' + output_text})
    completion = client.chat.completions.create(model = 'gpt-3.5-turbo',
                                            messages = gpt_messages,
                                            temperature = 0.5)
    # completion = json.loads(completion)
    return completion.choices[0].message.content

def main(args): 
    """Run a command line version of the generation and detection operations
        and optionally launch and serve the gradio demo"""
    # Initial arg processing and log
    args.normalizers = (args.normalizers.split(",") if args.normalizers else [])

    model, tokenizer, device = load_model(args)

    gpt_model_name = "D:\DDA4210\gpt"
    gptmodel = AutoModelForSeq2SeqLM.from_pretrained(gpt_model_name)
    gpttokenizer = AutoTokenizer.from_pretrained(gpt_model_name)    

    # with open("c4-train.00000-of-00512.json", "r", encoding='utf-8') as f:
    #     prompts_data = [json.loads(line) for line in f]
    with open("lfqa.json", "r", encoding='utf-8') as f:
        prompts_data = json.load(f)
    sample_idx = 1 # choose one prompt
    input_text = prompts_data[sample_idx]['title']
    args.default_prompt =input_text
    # print(input_text)

    # decoded_output_without_watermark, decoded_output_with_watermark= generate(input_text, 
    #                                                                                     args, 
    #                                                                                     model=model, 
    #                                                                                     device=device, 
    #                                                                                     tokenizer=tokenizer)
    # without_watermark_detection_result = detect(decoded_output_without_watermark, 
    #                                             args, 
    #                                             device=device, 
    #                                             tokenizer=tokenizer)
    # with_watermark_detection_result = detect(decoded_output_with_watermark, 
    #                                         args, 
    #                                         device=device,
    #                                         model = model,
    #                                         tokenizer=tokenizer)
    # ppl_without_watermark = compute_ppl(decoded_output_without_watermark, 
    #                                       args,
    #                                       model=gptmodel,
    #                                       device=device, 
    #                                       tokenizer=gpttokenizer)
    # print(decoded_output_with_watermark)
    # ppl_with_watermark = compute_ppl(decoded_output_with_watermark,
    #                                 args,
    #                                 model=gptmodel,
    #                                 device=device, 
    #                                 tokenizer=gpttokenizer)
    
    # rewritten_watermark_result = attack(decoded_output_with_watermark)
    # print(rewritten_watermark_result)
    # rewritten_with_watermark_detection_result = detect(rewritten_watermark_result, 
    #                                         args, 
    #                                         device=device, 
    #                                         model = model,
    #                                         tokenizer=tokenizer)
    # ppl_with_rewriten_watermark = compute_ppl(rewritten_watermark_result,
    #                                 args,
    #                                 model=gptmodel,
    #                                 device=device, 
    #                                 tokenizer=gpttokenizer)
    # print("generated text: " + decoded_output_with_watermark)
    # print(ppl_with_watermark)
    # print("rewrited text: " + rewritten_watermark_result)
    # print(ppl_with_rewriten_watermark)

    # print("Output without watermark:")
    # print(decoded_output_without_watermark)
    # print(f"Detection result @ {args.detection_z_threshold}:")
    # print(without_watermark_detection_result)

    # print(f"Detection result @ {args.detection_z_threshold}:")
    # print(with_watermark_detection_result)
    # print(rewritten_with_watermark_detection_result)

    return

if __name__ == "__main__":
    args = parse_args()
    main(args)