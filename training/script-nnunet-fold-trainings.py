import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description="Automate nnUNetv2 training for multiple datasets and plans.")
    parser.add_argument('--datasets', nargs='+', required=True, help='List of dataset identifiers')
    parser.add_argument('--configuration', required=True, default='3d_fullres', help='Configuration (e.g., 3d_fullres)')
    parser.add_argument('--trainer', required=True, nargs='+', help='Trainer (e.g., nnUNetTrainerNoDA)')
    parser.add_argument('--plans', nargs='+', required=True, help='List of plans')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode to print commands without executing them')
    parser.add_argument('--start_fra_fold', type=int, help="If flag set, restart the training from fold #N which is provided by the user with this flag.")

    args = parser.parse_args()

    start_fold = args.start_fra_fold if args.start_fra_fold is not None else 0

    for dataset in args.datasets:
        for plan in args.plans:
            for trainer in args.trainer:
                for fold in range(start_fold, 5):
                    run_training(dataset, args.configuration, trainer, fold, plan, args.debug)

    print("All trainings have been completed.")

def run_training(dataset, configuration, trainer, fold, plan, debug=False):

    command = ["nnUNetv2_train", dataset, configuration, str(fold), "--npz", "-p", plan, "-tr", trainer, "--val_best"]
    print('|')
    print(f"Executing command: {' '.join(command)}")
    print('|')

    if debug:
        print(f"Debug mode: {command}")
    else:
        try:
            print('||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||')
            print(f"Starting training for dataset {dataset}, fold {fold}, plan {plan}")
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            print(f"Finished training for dataset {dataset}, fold {fold}, plan {plan}")
            print('||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||')
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print('||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||')
            print(f"~~~~~~!!!!!!!!!!!Training for dataset {dataset}, fold {fold}, plan {plan} FAILED with error: {e.stderr}!!!!!!!!!!!!~~~~~~~")
            print('||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||')

if __name__ == "__main__":
    main()
