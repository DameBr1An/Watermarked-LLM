import json

from torch import tensor
import run
from transformers import (AutoTokenizer,
                          AutoModelForCausalLM,
                          AutoModelForSeq2SeqLM)

from argparse import Namespace

def ana(gamma,delta):
    args = Namespace()
    args.model_name_or_path="D:\\DDA4210\\facebookopt-1.3b"
    args.use_gpu=True
    args.prompt_max_length = None
    args.max_new_tokens=200
    args.gamma=gamma
    args.delta=delta
    args.detection_z_threshold=4.0
    args.generation_seed=42
    args.use_sampling=True
    args.sampling_temp=0.7
    args.n_beams=1
    args.normalizers=""
    args.ignore_repeated_ngrams=False

    model, tokenizer, device = run.load_model(args)
    gptmodel = AutoModelForSeq2SeqLM.from_pretrained("D:\\DDA4210\\gpt")
    gpttokenizer = AutoTokenizer.from_pretrained("D:\\DDA4210\\gpt")
    device = "cpu"
    # with open("c4-train.00000-of-00512.json", "r", encoding='utf-8') as f:
    #     prompts_data = [json.loads(line) for line in f]
    with open("lfqa.json", "r", encoding='utf-8') as f:
        prompts_data = json.load(f)
    sample_idx = 16  # choose one prompt
    input_text = prompts_data[sample_idx]['title']
    args.default_prompt =input_text

    analysis = {}
    _, _, decoded_output_without_watermark, decoded_output_with_watermark, _ = run.generate(input_text, 
                                                                                        args, 
                                                                                        model=model, 
                                                                                        device=device, 
                                                                                        tokenizer=tokenizer)
    ppl_without_watermark = run.compute_ppl(decoded_output_without_watermark, 
                                        args,
                                        model=gptmodel,
                                        device=device, 
                                        tokenizer=gpttokenizer)
    # print(decoded_output_with_watermark)
    ppl_with_watermark = run.compute_ppl(decoded_output_with_watermark,
                                    args,
                                    model=gptmodel,
                                    device=device, 
                                    tokenizer=gpttokenizer)
    analysis['ppl_without_watermark'] = ppl_without_watermark
    analysis['ppl_with_watermark'] = ppl_with_watermark
    analysis['gamma'] = args.gamma
    analysis['delta'] = args.delta
    analysis['z_threshold'] = args.detection_z_threshold
    without_watermark_detection_result = run.detect(decoded_output_without_watermark, 
                                                args, 
                                                device=device, 
                                                tokenizer=tokenizer)
    with_watermark_detection_result = run.detect(decoded_output_with_watermark, 
                                            args, 
                                            device=device, 
                                            tokenizer=tokenizer)
    analysis['T'] = with_watermark_detection_result[0][1]
    analysis['z'] = with_watermark_detection_result[3][1]
    analysis['p'] = with_watermark_detection_result[4][1]
    analysis['prediction'] = with_watermark_detection_result[6][1]
    analysis['confidence'] = with_watermark_detection_result[7][1]
    rewritten_watermark_result = run.attack(decoded_output_with_watermark)
    rewritten_with_watermark_detection_result = run.detect(rewritten_watermark_result, 
                                            args, 
                                            device=device, 
                                            tokenizer=tokenizer)
    analysis['attack_T'] = rewritten_with_watermark_detection_result[0][1]
    analysis['attack_z'] = rewritten_with_watermark_detection_result[3][1]
    analysis['attack_p'] = rewritten_with_watermark_detection_result[4][1]
    analysis['attack_prediction'] = rewritten_with_watermark_detection_result[6][1]
    # analysis['attack_confidence'] = rewritten_with_watermark_detection_result[7][1]

    return analysis

print(ana(gamma = 0.25, delta = 2.0))
