import time
import utils
from argparse import Namespace

def ana():
    args = Namespace()
    args.generate_model="D:\\DDA4210\\facebookopt-1.3b"
    args.util_model="D:\\DDA4210\\gpt"
    args.prompts_name="lfqa.json"
    args.prompt_index = 10
    args.use_gpu=True
    args.prompt_max_length = None
    args.max_new_tokens=200
    args.gamma=0.25
    args.delta=2
    args.detection_z_threshold=4.0
    args.generation_seed=42
    args.load_fp16 = False

    model, tokenizer, device, pplmodel, ppltokenizer = utils.load_model(args)

    for gamma in [0.25]:
        for delta in [0.5,1,2,5,10]:
            for T in range(5,200,10):
                for i in range(5):
                    args.gamma=gamma
                    args.delta=delta
                    args.max_new_tokens=T+5
                    args.min_new_tokens=T-5
                    input_text = utils.load_prompts()
                    without_wm, with_wm= utils.generate(input_text, 
                                            args, 
                                            model=model, 
                                            device=device, 
                                            tokenizer=tokenizer)

                    with_wm_detection = utils.detect(with_wm, 
                                                    args, 
                                                    device=device, 
                                                    model = model,
                                                    tokenizer=tokenizer)

                    ppl_with_wm = utils.compute_ppl(with_wm,
                                                    args,
                                                    model=pplmodel,
                                                    device=device, 
                                                    tokenizer=ppltokenizer)

                    analysis = {}
                    analysis['gamma'] = args.gamma
                    analysis['delta'] = args.delta
                    analysis['T'] = float(with_wm_detection[0][1]) / float(with_wm_detection[1][1][:-1]) * 100
                    analysis['z_with_watermark'] = with_wm_detection[2][1]
                    analysis['ppl_with_watermark'] = ppl_with_wm

                    with open("vis_fix_gamma.txt", 'a') as file:
                        print(analysis)
                        file.write(str(gamma) + ' ')
                        file.write(str(delta) + ' ')
                        file.write(str(analysis['T']) + ' ')
                        file.write(str(analysis['z_with_watermark']) + ' ')
                        file.write(str(float(analysis['ppl_with_watermark'])) + '\n')

ana()
