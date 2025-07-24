import argparse
import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader

import data
from metrics import aar
import nets

def print_metrics(model_name, y_pred, y_true):
    """Prints the evaluation metrics."""
    mae_total = np.abs(y_true - y_pred).mean()
    print(f'\n--- Evaluation Results for {model_name} ---')
    print(f'Overall MAE: {mae_total:.3f}')
    
    AAR, MAE, *_, sigmas, maes = aar(y_true, y_pred)
    print(f'Overall AAR: {AAR:.3f}')
    
    # Format for clean printing
    maes_str = '\t'.join([f'{m:.3f}' for m in maes])
    sigmas_str = '\t'.join([f'{np.sqrt(s):.3f}' for s in sigmas])

    print("\n" + "="*40)
    print("Metric\t" + "\t".join([f"Bin {i}" for i in range(1, 9)]) + "\tOverall")
    print("-" * 40)
    print(f'MAE\t{maes_str}\t{MAE:.3f}')
    print(f'Std Dev\t{sigmas_str}\t-')
    print("="*40 + "\n")


def main(args):
    # --- Model Initialization ---
    print(f"Loading model: {args.model_name}")
    model_class = getattr(nets, args.model_name)
    model = model_class()

    # --- Load Checkpoint ---
    print(f"Loading checkpoint: {args.checkpoint_path}")
    ckpt = torch.load(args.checkpoint_path, map_location=torch.device('cpu'))
    
    # Handle different state_dict structures (e.g., for ViT)
    if 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
        # Special handling for ViT model keys
        if args.model_name == 'ViT':
            state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
    else:
        # Fallback for checkpoints that are just the state_dict
        model.load_state_dict(ckpt)

    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    # --- Data Loading ---
    print(f"Loading data from: {args.test_csv}")
    valset = data.GTADataset(
        args.test_csv, 
        args.img_path,
        transform=data.EVAL_TRANSFORMS,
        return_paths=True
    )
    dataloader = DataLoader(
        valset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers
    )

    # --- Evaluation Loop ---
    N = len(valset)
    outs = np.zeros((N, 1))
    ages = np.zeros((N, 1))
    pbar = tqdm.tqdm(dataloader, desc="Evaluating")

    with torch.no_grad():
        for i, (imgs, age, paths) in enumerate(pbar):
            if torch.cuda.is_available():
                imgs = imgs.cuda()
            
            # Predict and de-normalize
            out = model(imgs) * 81
            out = out.detach().cpu().numpy()
            out = np.round(out).astype('int').clip(1, None)

            # Store results
            idx = i * args.batch_size
            batch_len = len(age)
            outs[idx:idx + batch_len, :] = out
            ages[idx:idx + batch_len, :] = age.reshape(-1, 1)

            # Update progress bar with running metrics
            # FIX: Flatten the `out` array to ensure its shape matches the `age` array.
            batch_aar, batch_mae, *_ = aar(age.numpy(), out.flatten())
            pbar.set_description(f'Batch AAR: {batch_aar:.3f} | Batch MAE: {batch_mae:.3f}')

    # --- Final Metrics ---
    print_metrics(args.model_name, outs, ages)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate a trained age estimation model.")
    
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model class (e.g., "ResNext", "ViT").')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the model .ckpt file.')
    
    parser.add_argument('--test_csv', type=str, default='data/test.csv', help='Path to the test data CSV file.')
    parser.add_argument('--img_path', type=str, default='/cluster/VAST/civalab/results/guess-the-age/training_caip_contest', help='Path to the directory containing test images.')
    
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for validation.')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for the DataLoader.')

    args = parser.parse_args()
    main(args)
