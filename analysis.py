import time
import utils
from argparse import Namespace

def ana():
    args = Namespace()
    args.generate_model="D:\\DDA4210\\facebookopt-1.3b"
    args.util_model="D:\\DDA4210\\gpt"
    args.prompts_name="c4-train.00000-of-00512.json"
    args.prompt_index = 22
    args.use_gpu=True
    args.detection_z_threshold=4.0
    args.generation_seed=42
    args.max_new_tokens=100
    model, tokenizer, device, pplmodel, ppltokenizer = utils.load_model(args)

    original_answer, input_text = utils.load_prompts()
    # input_text = utils.load_prompts(args)

    for gamma in [0.1,0.25,0.5,0.75,0.9]:
        for delta in [0,1,2,5,10]:
            for T in [25, 50, 75, 100, 125, 150]:
                args.prompt_max_length = T
                args.gamma=gamma
                args.delta=delta


    # print('prompt: ' + input_text)
    # start_time = time.time()
                without_wm, with_wm= utils.generate(input_text, 
                                        args, 
                                        model=model, 
                                        device=device, 
                                        tokenizer=tokenizer)
    # end_time = time.time()

    # 计算时间差
    # execution_time = end_time - start_time
    # print(f"generation finished in: {execution_time} seconds")
    # paraphrasing_wm = utils.paraphrasing_attack(with_wm)
    # substitution_wm = utils.substitution_attack(with_wm)
    # print('attack finished')
    # print(with_wm)
    # print(substitution_wm)
    # refined_wm = utils.refine(with_wm)
    # print('refine finished')
    # origin_detection = utils.detect(original_answer, 
    #                                 args, 
    #                                 device=device, 
    #                                 model = model,
    #                                 tokenizer=tokenizer)
    # without_wm_detection = utils.detect(without_wm, 
    #                                     args, 
    #                                     device=device, 
    #                                     model = model,
    #                                     tokenizer=tokenizer)
                with_wm_detection = utils.detect(with_wm, 
                                                args, 
                                                device=device, 
                                                model = model,
                                                tokenizer=tokenizer)
    # rewritten_with_wm_detection = utils.detect(paraphrasing_wm, 
    #                                         args, 
    #                                         device=device, 
    #                                         model = model,
    #                                         tokenizer=tokenizer)
    # print('detect finished')
    # ppl_original = utils.compute_ppl(original_answer, 
    #                                     args,
    #                                     model=pplmodel,
    #                                     device=device, 
    #                                     tokenizer=ppltokenizer)
    # ppl_without_wm = utils.compute_ppl(without_wm, 
    #                                     args,
    #                                     model=pplmodel,
    #                                     device=device, 
    #                                     tokenizer=ppltokenizer)
                ppl_with_wm = utils.compute_ppl(with_wm,
                                                args,
                                                model=pplmodel,
                                                device=device, 
                                                tokenizer=ppltokenizer)
    # ppl_rewritten_with_wm = utils.compute_ppl(paraphrasing_wm,
    #                                 args,
    #                                 model=pplmodel,
    #                                 device=device, 
    #                                 tokenizer=ppltokenizer)
                # print('compute perplexity finished')
                analysis = {}
                analysis['gamma'] = args.gamma
                analysis['delta'] = args.delta
                analysis['T'] = args.prompt_max_length
                # analysis['z_threshold'] = args.detection_z_threshold

                # analysis['T_with_watermark'] = with_wm_detection[0][1]
                analysis['z_with_watermark'] = with_wm_detection[2][1]
    # analysis['p_with_watermark'] = with_wm_detection[3][1]
    # analysis['watermark_words'] = with_wm_detection[4][1]
    # analysis['prediction_with_watermark'] = with_wm_detection[6][1]
    # analysis['confidence_with_watermark'] = with_wm_detection[7][1]
                analysis['ppl_with_watermark'] = ppl_with_wm

    # analysis['T_origin'] = origin_detection[0][1]
    # analysis['z_origin'] = origin_detection[2][1]
    # analysis['p_origin'] = origin_detection[3][1]
    # analysis['prediction_origin'] = origin_detection[6][1]
    # analysis['ppl_origin'] = ppl_original

    # analysis['T_without_watermark'] = without_wm_detection[0][1]
    # analysis['z_without_watermark'] = without_wm_detection[2][1]
    # analysis['p_without_watermark'] = without_wm_detection[3][1]
    # analysis['prediction_without_watermark'] = without_wm_detection[6][1]
    # analysis['ppl_without_watermark'] = ppl_without_wm

    # analysis['T_attack'] = rewritten_with_wm_detection[0][1]
    # analysis['z_attack'] = rewritten_with_wm_detection[2][1]
    # analysis['p_attack'] = rewritten_with_wm_detection[3][1]
    # analysis['prediction_attack'] = rewritten_with_wm_detection[6][1]
    # analysis['ppl_attack'] = ppl_rewritten_with_wm


    # print('#######################################')
    # print('watermark words:' , with_wm_detection[4][1])
    # print('generated with watermark: ' + with_wm)
    # print('#######################################')
    # print('generated without watermark: ' + without_wm)
    # print('#######################################')
    # print('refined with watermark: ' + refined_wm)
                with open("visual.txt", 'a') as file:
                    print(analysis)
                    file.write(str(gamma) + ' ')
                    file.write(str(delta) + ' ')
                    file.write(str(T) + ' ')
                    file.write(str(analysis['z_with_watermark']) + '\n')
    # return analysis

ana()
