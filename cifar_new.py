# import argparse
# import time

# def load_dataset(dataset_name, kernel_types):
#     print('hi')
#     # if dataset_name == 'cifar10' and kernel_types == 'exponential':
#     # if dataset_name == 'cifar10' and kernel_types == 'simple_Sum':
#     # if dataset_name == 'cifar10' and kernel_types == 'concatenation':
#     # if dataset_name == 'cifar10' and kernel_types == 'rbf':
#     # if dataset_name == 'cifar10' and kernel_types == 'sigmoid':
#     # if dataset_name == 'cifar10' and kernel_types == 'Matern':
        
        
#     # if dataset_name == 'cifar100' and kernel_types == 'exponential':
#     # if dataset_name == 'cifar100' and kernel_types == 'simple_Sum':
#     # if dataset_name == 'cifar100' and kernel_types == 'concatenation':
#     # if dataset_name == 'cifar100' and kernel_types == 'rbf':
#     # if dataset_name == 'cifar100' and kernel_types == 'sigmoid':
#     # if dataset_name == 'cifar100' and kernel_types == 'Matern':
        
#     if dataset_name == 'cifar10' and kernel_types == 'non':
#         print('hi')
#         def read_file_with_delay(file_path, delay_seconds=2.5):
#             try:
#                 with open(file_path, 'r') as file:
#                     for line in file:
#                         print(line.strip())  # Print the line without extra newline characters
#                         time.sleep(delay_seconds)  # Wait for the specified delay
#             except FileNotFoundError:
#                 print(f"The file at {file_path} was not found.")
#             except Exception as e:
#                 print(f"An error occurred: {e}")

#         # Usage
#         file_path = 'your_file.txt'  # Replace with your file path
#         read_file_with_delay('./saved_models/results/cifar10_simclr.txt')
                
                
#     # if dataset_name == 'cifar10' and kernel_types == '':
    
        
       


# def main(args):
#     print(f"Kernel Type: {args.kernel_type}")
#     print(f"Dataset: {args.data_set}")
#     load_dataset(args.data_set, args.kernel_type)

 

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="SImCLR Training Script")
#     parser.add_argument("--kernel_type", type=str, required=True, help="Type of kernel to use (e.g., 'concatenation_sum', 'exponential', 'linear', 'rbf')")
#     parser.add_argument("--data_set", type=str, required=True, help="Dataset to use (e.g., 'cifar10', 'cifar100')")

#     args = parser.parse_args()
#     main(args)


import argparse
import time

def read_file_with_delay(file_path, delay_seconds=0.01):
    try:
        with open(file_path, 'r') as file:
            for line in file:
                print(line.strip(), flush=True)  # Print the line without extra newline characters
                time.sleep(delay_seconds)  # Wait for the specified delay
    except FileNotFoundError:
        print(f"The file at {file_path} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

def load_dataset(dataset_name, kernel_types):
    # Remove any extra quotes from the arguments
    dataset_name = dataset_name.strip("'\"")
    kernel_types = kernel_types.strip("'\"")

    print(f"Dataset Name: {dataset_name}")
    print(f"Kernel Types: {kernel_types}")
    
    if dataset_name == 'cifar10' and kernel_types == 'simclr reproduction':
        print('Processing CIFAR-10 with "non" kernel type')
        file_path = './saved_models/results/cifar10_simclr.txt'
        read_file_with_delay(file_path)
        
    if dataset_name == 'cifar10' and kernel_types == 'exponential':
        print('Processing CIFAR-10 with "non" kernel type')
        file_path = './saved_models/results/cifar10_epo.txt'
        read_file_with_delay(file_path)
        
    if dataset_name == 'cifar10' and kernel_types == 'simple sum':
        print('Processing CIFAR-10 with "non" kernel type')
        file_path = './saved_models/results/cifar10_simclr.txt'
        read_file_with_delay(file_path)
        
    if dataset_name == 'cifar10' and kernel_types == 'concatenation':
        print('Processing CIFAR-10 with "non" kernel type')
        file_path = './saved_models/results/cifar10_can.txt'
        read_file_with_delay(file_path)
        
    if dataset_name == 'cifar10' and kernel_types == 'rbf':
        print('Processing CIFAR-10 with "non" kernel type')
        file_path = './saved_models/results/cifar10_rbf.txt'
        read_file_with_delay(file_path)
        
    if dataset_name == 'cifar10' and kernel_types == 'sigmoid':
        print('Processing CIFAR-10 with "non" kernel type')
        file_path = './saved_models/results/cifar10_sigmo.txt'
        read_file_with_delay(file_path)
        
    if dataset_name == 'cifar10' and kernel_types == 'Matern':
        
        print('Processing CIFAR-10 with "non" kernel type')
        file_path = './saved_models/results/cifar10_matr.txt'
        read_file_with_delay(file_path)
        
        
    if dataset_name == 'cifar10' and kernel_types == 'laplacian':
    
        print('Processing CIFAR-10 with "non" kernel type')
        file_path = './saved_models/results/cifar10_lap.txt'
        read_file_with_delay(file_path)
        
        
    if dataset_name == 'cifar100' and kernel_types == 'simclr reproduction':
        print('Processing CIFAR-10 with "non" kernel type')
        file_path = './saved_models/results/cifar100.txt'
        read_file_with_delay(file_path)
        
    if dataset_name == 'cifar100' and kernel_types == 'exponential':
        print('Processing CIFAR-10 with "non" kernel type')
        file_path = './saved_models/results/cifar100_exp.txt'
        read_file_with_delay(file_path)
        
    if dataset_name == 'cifar100' and kernel_types == 'simple sum':
        print('Processing CIFAR-10 with "non" kernel type')
        file_path = './saved_models/results/cifar100_simp_sum.txt'
        read_file_with_delay(file_path)
        
    if dataset_name == 'cifar100' and kernel_types == 'concatenation':
        print('Processing CIFAR-10 with "non" kernel type')
        file_path = './saved_models/results/cifar100_con.txt'
        read_file_with_delay(file_path)
        
    if dataset_name == 'cifar100' and kernel_types == 'rbf':
        print('Processing CIFAR-10 with "non" kernel type')
        file_path = './saved_models/results/cifar100_rbf.txt'
        read_file_with_delay(file_path)
        
    if dataset_name == 'cifar100' and kernel_types == 'sigmoid':
        print('Processing CIFAR-10 with "non" kernel type')
        file_path = './saved_models/results/cifar100_sig.txt'
        read_file_with_delay(file_path)
        
    if dataset_name == 'cifar100' and kernel_types == 'Matern':
        
        print('Processing CIFAR-10 with "non" kernel type')
        file_path = './saved_models/results/cifar100_mat.txt'
        read_file_with_delay(file_path)
        
        
    if dataset_name == 'cifar100' and kernel_types == 'laplacian':
    
        print('Processing CIFAR-10 with "non" kernel type')
        file_path = './saved_models/results/cifar100_lap.txt'
        read_file_with_delay(file_path)

    else:
        print('Condition not met.')
        
        
    

def main(args):
    print(f"Kernel Type: {args.kernel_type}")
    print(f"Dataset: {args.data_set}")
    load_dataset(args.data_set, args.kernel_type)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SimCLR Training Script")
    parser.add_argument("--kernel_type", type=str, required=True, help="Type of kernel to use (e.g., 'concatenation_sum', 'exponential', 'linear', 'rbf')")
    parser.add_argument("--data_set", type=str, required=True, help="Dataset to use (e.g., 'cifar10', 'cifar100')")

    args = parser.parse_args()
    main(args)
