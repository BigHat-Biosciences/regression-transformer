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

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--search_method", type=str, default="sample", choices=["sample", "beam", "greedy", "denoise", "gibbs"])
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--tolerance", type=float, default=None, help="The tolerance for the property goal")
    parser.add_argument("--num_avg_mutations", type=int, default=6., help="The number of average mutations to make")
    parser.add_argument("--num_samples", type=int, default=1000, help="The number of samples to generate")
    parser.add_argument("--conditioning_value", type=float, nargs="+", default=[0.5], help="The value to condition on")
    parser.add_argument("--property_tokens", type=str, nargs="+", default=["<tm>"], help="The property tokens to condition on")

    args = parser.parse_args()
    print(args)

    lead_seq = 'KVQLVESGGGVVQPGGSLRLSCAASGFSFRNFGMSWVRQAPGKGPEWVSAISGSGADTLYASPVKGRFIISRDNAKNTLYLQMNSLRPEDTAVYYCTIGGSLTRSSQGTLVTVSS'
    mutable_res_mask = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, False, True, True, True, True, False, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
    cdr_mask = [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False]

    assert len(args.property_tokens) == len(args.conditioning_value), "Number of property tokens must match number of conditioning values"

    conditional_generator = ProteinLanguageRT(
        resources_path=args.model_path,
        context=lead_seq,
        search=args.search_method,
        temperature=args.temperature,
        batch_size=args.batch_size,
        device=device,
        tolerance=args.tolerance,
        sampling_wrapper={
            'property_goal': { p: v for p, v in zip(args.property_tokens, args.conditioning_value) },
            'fraction_to_mask': args.num_avg_mutations/len(lead_seq),
            'masking_strategy': 'stochastic',
            'bool_context_mask': mutable_res_mask,
            'allowed_sampling_tokens': 'ACDEFGHIKLMNPQRSTVWY'
        },
        inference_config={
            "property_token": args.property_tokens,
            "property_ranges": { p: [0., 1.] for p in args.property_tokens },
            "max_span_length": 7,
            "property_mask_length": { p: 5 for p in args.property_tokens }
        }
    )
    
    ######## Sample from the model ########
    os.environ["PARTNER"] = "capulet"
    os.environ["DEPLOYMENT_ENVIRONMENT"] = "prod"
    from conditional_plm.oracles import ThermoOracle
    from conditional_plm.data.capulet import get_capulet_reference_sequence
    from conditional_plm.data.humanness import biophi_v_humannesses, DEFAULT_MIN_PERCENT_SUBJECTS

    samples = set()

    for batch_idx in tqdm(range(args.num_samples // args.batch_size)):
        new_samples = conditional_generator.generate_batch(lead_seq)
        new_sequences = [sample[0] for sample in new_samples]
        sample_properties = [parse_properties(sample[1], tags=args.property_tokens) for sample in new_samples]
        sample_idx_to_keep = [i for i in range(len(new_sequences)) if None not in sample_properties[i]]
        good_samples = [(new_sequences[i], sample_properties[i]) for i in sample_idx_to_keep]
        if good_samples:
            print([sample[1] for sample in good_samples])
            samples.update(good_samples)

    samples = list(samples)
    print(f"Generated {len(samples)} samples")
    for i, prop in enumerate(args.property_tokens):
        print(f"Mean RT {prop}: {np.mean([sample[1][i] for sample in samples])}")
    
    ######## Compute properties ########
    samples = [sample[0] for sample in samples]
    therm_oracle = ThermoOracle.load_default()
    ref_seq = get_capulet_reference_sequence()

    tm_preds = therm_oracle.forward(samples, ref_seq)

    df = pd.DataFrame({'sequence': samples})
    df['tm_mean'] = tm_preds.cpu().detach().numpy()
    print(f"Mean TM: {np.mean(df['tm_mean'])}")

    biophi_objs = biophi_v_humannesses(samples)
    oasis_percentile = [obj.get_oasis_percentile(DEFAULT_MIN_PERCENT_SUBJECTS / 100) for obj in biophi_objs]
    df['oasis_percentile'] = oasis_percentile
    print(f"Mean OASIS percentile: {np.mean(df['oasis_percentile'])}")

    breakpoint()
    os.makedirs(args.output_dir, exist_ok=True)
    df.to_csv(os.path.join(args.output_dir, f"{args.run_name}.csv"), index=False)
