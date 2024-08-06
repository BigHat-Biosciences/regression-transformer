import argparse
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Any, Dict

device = "cuda" if torch.cuda.is_available() else "cpu"

from terminator.sampler import ProteinLanguageRT


def parse_property(property, tag="<tm>"):
    if tag not in property: return None
    try: return float(property.split(tag)[1].split('<')[0].strip())
    except: return None

def parse_properties(properties, tags=["<tm>"]):
    res = ()
    for tag in tags:
        res += (parse_property(properties, tag),)
    return res

def parse_sequence_for_pp(seq, props=["<tm>"]):
    for prop in props[::-1]:
        seq = f"{prop}[MASK][MASK][MASK][MASK][MASK]|"+seq
    return seq

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--samples_path", type=str, required=True, help="The .csv file containing the samples. Should have a 'text' column")
    parser.add_argument("--output_dir", type=str, required=True, help="The output directory to save the results")
    parser.add_argument("--property_tokens", type=str, nargs="+", default=["<tm>"], help="The property tokens to condition on")

    args = parser.parse_args()
    print(args)

    ########## Load the model and data ##########
    os.environ["PARTNER"] = "capulet"
    os.environ["DEPLOYMENT_ENVIRONMENT"] = "prod"
    from conditional_plm.data.capulet import get_capulet_reference_sequence

    ref_seq = get_capulet_reference_sequence()
    samples_df = pd.read_csv(args.samples_path)

    property_predictor = ProteinLanguageRT(
        resources_path=args.model_path,
        search="greedy",
        temperature=args.temperature,
        batch_size=args.batch_size,
        device=device,
        context=parse_sequence_for_pp(ref_seq, args.property_tokens),
        tolerance=100.,
        inference_config={
            "normalize": [False] * len(args.property_tokens),
            "property_token": args.property_tokens,
            "property_ranges": { p: [0., 1.] for p in args.property_tokens },
            "max_span_length": len(ref_seq),
            "property_mask_length": { p: 5 for p in args.property_tokens }
        }
    )
    
    ######## Inference the model ########
    samples = samples_df['text'].apply(lambda x: parse_sequence_for_pp(x, args.property_tokens)).tolist()
    preds = property_predictor.generate_batch(samples)
    preds_by_prop = { tag: [parse_property(s, tag) for s in preds] for tag in args.property_tokens }
    
    ########## Extract the predictions ##########    
    for prop in args.property_tokens:
        samples_df[f"{prop}_pred"] = preds_by_prop[prop]

    os.makedirs(args.output_dir, exist_ok=True)
    samples_df.to_csv(os.path.join(args.output_dir, f"{args.run_name}.csv"), index=False)
