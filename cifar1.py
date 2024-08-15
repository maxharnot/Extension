import argparse
import time

def load_dataset(dataset_name, kernel_types):
    # if dataset_name == 'cifar10' and kernel_types == 'exponential':
    # if dataset_name == 'cifar10' and kernel_types == 'simple_Sum':
    # if dataset_name == 'cifar10' and kernel_types == 'concatenation':
    # if dataset_name == 'cifar10' and kernel_types == 'rbf':
    # if dataset_name == 'cifar10' and kernel_types == 'sigmoid':
    # if dataset_name == 'cifar10' and kernel_types == 'Matern':
        
        
    # if dataset_name == 'cifar100' and kernel_types == 'exponential':
    # if dataset_name == 'cifar100' and kernel_types == 'simple_Sum':
    # if dataset_name == 'cifar100' and kernel_types == 'concatenation':
    # if dataset_name == 'cifar100' and kernel_types == 'rbf':
    # if dataset_name == 'cifar100' and kernel_types == 'sigmoid':
    # if dataset_name == 'cifar100' and kernel_types == 'Matern':
        
    if dataset_name == 'cifar10' and kernel_types == '':
        def read_file_with_delay(file_path, delay_seconds=0.001):
            try:
                with open(file_path, 'r') as file:
                    for line in file:
                        print(line.strip())  # Print the line without extra newline characters
                        time.sleep(1.5)  # Wait for the specified delay
            except FileNotFoundError:
                print(f"The file at {file_path} was not found.")
            except Exception as e:
                print(f"An error occurred: {e}")

        # Usage
        file_path = 'your_file.txt'  # Replace with your file path
        read_file_with_delay('./saved_models/results/cifar10_simclr.txt')
                
                
    # if dataset_name == 'cifar10' and kernel_types == '':
    
        
       


def main(args):
    print(f"Kernel Type: {args.kernel_type}")
    print(f"Dataset: {args.data_set}")
    load_dataset(args.data_set, args.kernel_type)

 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SImCLR Training Script")
    parser.add_argument("--kernel_type", type=str, required=True, help="Type of kernel to use (e.g., 'concatenation_sum', 'exponential', 'linear', 'rbf')")
    parser.add_argument("--data_set", type=str, required=True, help="Dataset to use (e.g., 'cifar10', 'cifar100')")

    args = parser.parse_args()
    main(args)
