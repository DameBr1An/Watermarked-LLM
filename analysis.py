import json
import run
from transformers import (AutoTokenizer,
                          AutoModelForSeq2SeqLM,
                          GPT2LMHeadModel,
                          GPT2Tokenizer)
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
    pplmodel = GPT2LMHeadModel.from_pretrained("D:\\DDA4210\\gpt")
    ppltokenizer = GPT2Tokenizer.from_pretrained("D:\\DDA4210\\gpt")
    device = "cpu"
    # with open("c4-train.00000-of-00512.json", "r", encoding='utf-8') as f:
    #     prompts_data = [json.loads(line) for line in f]
    with open("lfqa.json", "r", encoding='utf-8') as f:
        prompts_data = json.load(f)
    sample_idx = 66  # choose one prompt
    input_text = prompts_data[sample_idx]['title']
    best_score = max(prompts_data[sample_idx]["answers"]["score"])
    answer_index = prompts_data[sample_idx]["answers"]["score"].index(best_score)
    args.original_answer = prompts_data[sample_idx]["answers"]["text"][answer_index]
    args.default_prompt =input_text

    analysis = {}
    without_wm, with_wm= run.generate(input_text, 
                                                args, 
                                                model=model, 
                                                device=device, 
                                                tokenizer=tokenizer)
    print('#######################################')
    print(with_wm)
    print('#######################################')
    print(without_wm)
    
    rewritten_wm = run.attack(with_wm)
    origin_detection = run.detect(args.original_answer, 
                                    args, 
                                    device=device, 
                                    model = model,
                                    tokenizer=tokenizer)
    without_wm_detection = run.detect(without_wm, 
                                        args, 
                                        device=device, 
                                        model = model,
                                        tokenizer=tokenizer)
    with_wm_detection = run.detect(with_wm, 
                                    args, 
                                    device=device, 
                                    model = model,
                                    tokenizer=tokenizer)
    rewritten_with_wm_detection = run.detect(rewritten_wm, 
                                            args, 
                                            device=device, 
                                            model = model,
                                            tokenizer=tokenizer)
    ppl_original = run.compute_ppl(args.original_answer, 
                                        args,
                                        model=pplmodel,
                                        device=device, 
                                        tokenizer=ppltokenizer)
    ppl_without_wm = run.compute_ppl(without_wm, 
                                        args,
                                        model=pplmodel,
                                        device=device, 
                                        tokenizer=ppltokenizer)
    ppl_with_wm = run.compute_ppl(with_wm,
                                    args,
                                    model=pplmodel,
                                    device=device, 
                                    tokenizer=ppltokenizer)
    ppl_rewritten_with_wm = run.compute_ppl(rewritten_wm,
                                    args,
                                    model=pplmodel,
                                    device=device, 
                                    tokenizer=ppltokenizer)
    
    analysis['gamma'] = args.gamma
    analysis['delta'] = args.delta
    analysis['z_threshold'] = args.detection_z_threshold

    analysis['T_with_watermark'] = with_wm_detection[0][1]
    analysis['z_with_watermark'] = with_wm_detection[2][1]
    analysis['p_with_watermark'] = with_wm_detection[3][1]
    analysis['watermark_words'] = with_wm_detection[4][1]
    analysis['prediction_with_watermark'] = with_wm_detection[6][1]
    # analysis['confidence_with_watermark'] = with_wm_detection[7][1]
    analysis['ppl_with_watermark'] = ppl_with_wm

    analysis['T_origin'] = origin_detection[0][1]
    analysis['z_origin'] = origin_detection[2][1]
    analysis['p_origin'] = origin_detection[3][1]
    analysis['prediction_origin'] = origin_detection[6][1]
    analysis['ppl_origin'] = ppl_original

    analysis['T_without_watermark'] = without_wm_detection[0][1]
    analysis['z_without_watermark'] = without_wm_detection[2][1]
    analysis['p_without_watermark'] = without_wm_detection[3][1]
    analysis['prediction_without_watermark'] = without_wm_detection[6][1]
    analysis['ppl_without_watermark'] = ppl_without_wm

    analysis['T_attack'] = rewritten_with_wm_detection[0][1]
    analysis['z_attack'] = rewritten_with_wm_detection[2][1]
    analysis['p_attack'] = rewritten_with_wm_detection[3][1]
    analysis['prediction_attack'] = rewritten_with_wm_detection[6][1]
    analysis['ppl_attack'] = ppl_rewritten_with_wm

    return analysis

print(ana(gamma = 0.25, delta = 2.0))
