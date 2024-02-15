import os
import torch
import json
import argparse
from torchvision.utils import save_image
from pytorch_lightning import seed_everything
import torch.nn.functional as F
from torchvision.io import read_image

from dreammatcher.diffuser_utils import (
    DreamMatcherPipeline,
    CustomDiffusionPipeline,
)
from dreammatcher.dreammatcher_utils import AttentionBase, regiter_attention_editor_diffusers
from dreammatcher.dreammatcher import (
    MutualSelfAttentionControlMaskAuto_Matching,
)

from diffusers import DDIMScheduler

def load_image(image_path, device):
    image = read_image(image_path)
    image = image[:3].unsqueeze_(0).float() / 127.5 - 1.0  # [-1, 1]
    image = F.interpolate(image, (512, 512))
    image = image.to(device)
    return image

def main(params, p_idx):
    # cuda settings
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if params["dtype"] == "float16":
        torch_dtype = torch.float16
        dtype = torch.float16
    elif params["dtype"] == "float32":
        torch_dtype = torch.float32
        dtype = torch.float32
    else:
        torch_dtype = None
        dtype = torch.float32

    # set prompts
    prompts = [
        params["ref_prompt"].replace("<", "\u003C"),
        params["gen_prompt"].replace("<", "\u003C"),
    ]  # @param {type:"string"}
    print(f"REF PROMPT : {prompts[0]} \nGEN PROMPT : {prompts[1]}")

    # load source image
    SOURCE_IMAGE_PATH = params["invert_path"]
    source_image = load_image(SOURCE_IMAGE_PATH, device)

    # Set the output directories
    prompt_out_dir = os.path.join(params["out_dir"], f"output/ours/prompt_{p_idx}")
    os.makedirs(prompt_out_dir, exist_ok=True)  # output directory

    if params["save_pred_x0"]:
        pred_x0_dir_src = os.path.join(prompt_out_dir, "vis/pred_x0/src")
        pred_x0_dir_trg = os.path.join(prompt_out_dir, "vis/pred_x0/trg")
        pred_x0_dir_cond_trg = os.path.join(prompt_out_dir, "vis/pred_x0/trg_cond")
        pred_x0_dir_uncond_trg = os.path.join(prompt_out_dir, "vis/pred_x0/trg_uncond")
        os.makedirs(pred_x0_dir_src, exist_ok=True)  # pred_x0 directory (src)
        os.makedirs(pred_x0_dir_trg, exist_ok=True)  # pred_x0 directory (trg)
        os.makedirs(pred_x0_dir_cond_trg, exist_ok=True)  # pred_x0 directory (cond trg)
        os.makedirs(pred_x0_dir_uncond_trg, exist_ok=True)  # pred_x0 directory (uncond trg)

    if params["save_mask"]:
        mask_dir = os.path.join(prompt_out_dir, f"vis/mask")
        os.makedirs(mask_dir, exist_ok=True)  # foreground mask directory
    else:
        mask_dir = None
   
    if method == "custom_diffusion":
        scheduler = DDIMScheduler.from_pretrained(
        params["concept_dir"], subfolder="scheduler", torch_dtype=torch_dtype
    )

        model = CustomDiffusionPipeline.from_pretrained(
            params["concept_dir"],
            scheduler=scheduler,
            safety_checker=None,
            torch_dtype=torch_dtype,
        ).to(device)
        model.load_model(params["delta_ckpt"], False)
    
    elif method == "ti":
        scheduler = DDIMScheduler.from_pretrained(
        "CompVis/stable-diffusion-v1-4", subfolder="scheduler", torch_dtype=torch_dtype
    )

        model = DreamMatcherPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            scheduler=scheduler,
            safety_checker=None,
            torch_dtype=torch_dtype,
        ).to(device)

        model.load_textual_inversion(params["concept_dir"])
    else:
        scheduler = DDIMScheduler.from_pretrained(
        params["concept_dir"], subfolder="scheduler", torch_dtype=torch_dtype
    )

        model = DreamMatcherPipeline.from_pretrained(
            params["concept_dir"],
            scheduler=scheduler,
            safety_checker=None,
            torch_dtype=torch_dtype,
        ).to(device)

    # invert the source image
    # get inversion latent for consistent source image synthesis
    seed_everything(params["seed"])
    start_code, latents_list = model.invert(
        source_image.to(dtype),
        "",
        guidance_scale=params["guidance_scale"],
        num_inference_steps=params["inference_steps"],
        return_intermediates=True,
    )

    num_samples = params.get("num_samples", 1)

    # random end_code for diverse target image synthesis
    end_code = torch.randn([num_samples, 4, 64, 64], device=device, dtype=dtype)

    # Baseline: textual inversion, dreambooth, custom diffusion
    print("Baseline...")
    for sampling_idx in range(num_samples):
        print(f"[Baseline] Sampling.. ({sampling_idx + 1} / {num_samples})")
        editor = AttentionBase()
        regiter_attention_editor_diffusers(model, editor)
        initial_code = torch.cat(
            [start_code, end_code[sampling_idx].unsqueeze(0)], dim=0
        )
        seed_everything(params["seed"])
        model_output = model(
            prompts,
            latents=initial_code,
            num_inference_steps=params["inference_steps"],
            guidance_scale=params["guidance_scale"],
            outdir=prompt_out_dir,
            save_pred_x0=False,
        )
        out_path = os.path.join(
            params["out_dir"],
            "output",
            "baseline",
            f"prompt_{p_idx}",
            f"{sampling_idx}.jpg",
        )
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        save_image(model_output[1], out_path)

    # Ours: baseline + matching
    print("Baseline + Matching...")
    mg_params = params if params["matching_guidance"] else None
    ref_latents = latents_list if params["use_inversion_latents"] else None
    for sampling_idx in range(num_samples):
        print(f"[Ours] Sampling.. ({sampling_idx + 1} / {num_samples})")

        editor = MutualSelfAttentionControlMaskAuto_Matching(
            params, mask_save_dir=mask_dir, save_dir=prompt_out_dir
        )
        regiter_attention_editor_diffusers(model, editor)
        initial_code = torch.cat(
            [start_code, end_code[sampling_idx].unsqueeze(0)], dim=0
        )
        seed_everything(params["seed"])
        model_output = model(
            prompts,
            latents=initial_code,
            num_inference_steps=params["inference_steps"],
            guidance_scale=params["guidance_scale"],
            ref_intermediate_latents=ref_latents,
            matching_guidance=mg_params,
            outdir=prompt_out_dir,
            save_pred_x0=params["save_pred_x0"],
        )
        out_path = os.path.join(
            params["out_dir"],
            "output",
            "ours",
            f"prompt_{p_idx}",
            f"{sampling_idx}.jpg",
        )
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        save_image(model_output[1], out_path)

    with open(
        os.path.join(params["out_dir"], f"prompt_{p_idx}_params.json"), "w"
    ) as file:
        json.dump(params, file, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a sampling scripts for the dreammatcher')
    parser.add_argument('--models', type=str, help='Name of personalized models to use', default='dreambooth', choices=["ti", "dreambooth", "custom_diffusion"])
    parser.add_argument('--num_device', type=int, help='number of device', default=0)
    parser.add_argument('--num_samples', type=int, help='number of samples', default=8)
    parser.add_argument('--mode', type=str, help='mode to evaluate', default="normal", choices=["normal", "challenging"])
    parser.add_argument('--result_dir', type=str, help='dir to result images', default="./results/")

    args = parser.parse_args()

    vico_concepts = [
        ("cat_statue", "cat-toy", False),
        ("elephant_statue", "elephant_statue", False),
        ("duck_toy", "duck_toy", False),
        ("monster_toy", "monster_toy", False),
        #####################
        ("brown_teddybear", "teddybear", False),
        ("tortoise_plushy", "tortoise_plushy", False),
        ("brown_dog", "brown_dog", True),
        ("fat_dog", "fat_dog", True),
        #####################
        ("brown_dog2", "brown_dog2", True),
        ("black_cat", "black_cat", True),
        ("brown_cat", "brown_cat", True),
        ("alarm_clock", "clock", False),
        #####################
        ("pink_sunglasses", "pink_sunglasses", False),
        ("red_teapot", "red_teapot", False),
        ("red_vase", "vase", False),
        ("wooden_barn", "barn", False),
    ]

    for concept, concept_orig_name, is_live in vico_concepts:
        if args.mode == "challenging":
            if is_live:
                with open("./inputs/prompts_live_objects_challenging.txt", "r") as fin:
                    prompts = fin.read()
                    prompts = prompts.split("\n")
            else:
                with open("./inputs/prompts_nonlive_objects_challenging.txt", "r") as fin:
                    prompts = fin.read()
                    prompts = prompts.split("\n")
        else:
            if is_live:
                with open("./inputs/prompts_live_objects.txt", "r") as fin:
                    prompts = fin.read()
                    prompts = prompts.split("\n")
            else:
                with open("./inputs/prompts_nonlive_objects.txt", "r") as fin:
                    prompts = fin.read()
                    prompts = prompts.split("\n")

        concept_model_name = f"{concept}"
        num_samples = args.num_samples
        method = args.models
        torch.cuda.set_device(args.num_device)  # set the GPU device

        assert method in [
            "ti",
            "dreambooth",
            "custom_diffusion",
        ], f"{method} was not implemented"

        if method == "ti":
            concept_dir = f"./concept_models/ti/ti-concept-diffusers-230905-embeds/{concept_model_name}/learned_embeds.bin"
            out_dir = f"{args.result_dir}/{concept_model_name}"
            delta_ckpt = None
        elif method == "dreambooth":
            concept_dir = f"./concept_models/dreambooth/dreambooth-concept-original-prior-230821/{concept_model_name}/checkpoints/diffusers"
            out_dir = f"{args.result_dir}/{concept_model_name}"
            delta_ckpt = None
        elif method == "custom_diffusion":
            concept_dir = "CompVis/stable-diffusion-v1-4"
            out_dir = f"{args.result_dir}/{concept_model_name}"
            delta_ckpt = (
                f"./concept_models/custom_diffusion/logs_diffusers/{concept}/delta.bin"
            )
        
        for p_idx, prompt in enumerate(prompts):
            if method == "ti":
                ref_prompt = f"a photo of a <{concept}>"
                ref_token_idx = 5
                gen_prompt = prompt.format(f"<{concept}>")
                cur_token_idx = (
                    gen_prompt.split(" ").index(f"<{concept}>") + 1
                )  # 1-indexed because of <sos>
            elif method == "dreambooth":
                ref_prompt = f"a photo of a sks {concept}"
                ref_token_idx = 6
                gen_prompt = prompt.format(f"sks {concept}")
                cur_token_idx = (
                    gen_prompt.split(" ").index(f"{concept}") + 1
                )  # 1-indexed because of <sos>
            elif method == "custom_diffusion":
                ref_prompt = f"a photo of a <new1> {concept}"
                ref_token_idx = 6
                gen_prompt = prompt.format(f"<new1> {concept}")
                cur_token_idx = (
                    gen_prompt.split(" ").index(f"{concept}") + 1
                )  # 1-indexed because of <sos>

            with open("./inputs/imgs_source.txt", "r") as file:
                src_imgs_dict = json.load(file)
                invert_path = os.path.join(
                    f"./inputs/{concept_orig_name}",
                    src_imgs_dict[f"{concept_orig_name}"],
                )

            params = {
                "seed": 42,
                "dtype": None,  # float16, float32, None(="auto")
                "ref_prompt": ref_prompt,
                "ref_token_idx": ref_token_idx,  # IMPORTANT: location of the placeholder_token, <cat-toy> here
                "gen_prompt": gen_prompt,
                "cur_token_idx": cur_token_idx,  # IMPORTANT: location of the placeholder_token, <cat-toy> here
                "concept_dir": concept_dir,
                "delta_ckpt": delta_ckpt,
                "out_dir": out_dir,
                "invert_path": invert_path,  # the source image
                "num_samples": num_samples,
                "guidance_scale": 10.0,
                "inference_steps": 50,
                "initial_step": 4,  # IMPORTANT: initial timestep of matching attention (recommendation: 4)
                "initial_layer": 10,  # IMPORTANT: initial timestep of matching attention (recommentation: 10)
                "cut_step": 50,  # IMPORTANT: last timestep of matching attention (recommendation: 50)
                "cut_layer": 16,  # IMPORTANT: last layer of matching attention (recommendation: 16)
                "thres": 0.05,  # threshold for cross-attention foreground mask (recommendation: 0.05)
                "cc_thres": 0.4,  # IMPORTANT: cycle consistency threshold (a larger threshold injects more reference appearance)
                "use_inversion_latents": True,
                "key_replace": False,  # recommendation: False
                "fg_mask": True,  # recommendation: True, matching between foregrounds
                "matching_guidance": False,  # use matching guidance
                "mg_initial_step": 4,  # IMPORTANT: matching guidance starting timestep
                "mg_cut_step": 50,  # matching guidance last timestep
                "mg_cc_thres": 0.4,  # matching guidance cycle consistency (recommendation: same as cc_thres)
                "mg_grad_weight": 50,  # important: matching guidance weight
                "save_pred_x0": False,  # for visualization, if you want to reduce time, set this false
                "save_mask": False,  # for visualization, if you want to reduce time, set this false
            }  

            main(params, p_idx)
