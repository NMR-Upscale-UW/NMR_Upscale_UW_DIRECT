### Taking Notes for PyTorch
----

## Installation:
go to:

https://pytorch.org/get-started/locally/

Select operating system and package manager

If want GPU support for deep learning models (must have Nvidia GPU locally)

https://developer.nvidia.com/cuda-downloads

- Make sure that the CUDA toolkit is compatible version

## Tensor Basics

Everything in PyTorch is works as a tensor

Tensors are very similar to matrices(flashback to ChemE 530). A lot of operator
as would be for numpy matrices. Tensors allow for higher order dimensions to be
represented as tensors.

- In fact, it is fairly simple to convert from NumPy array and a Torch tensor

#### To specify if an operatation on tensors is on cpu or gpu

if torch.cuda.is_available():
	device = torch.device('cuda')
	x = torch.ones(5,device=device)
	y = torch.ones(5)
	y = y.to(device)
	z = x + y
	z = z.to('cpu)

## Autograd

Gradient is a commonly used term in optimization and machine learning.

If we are to use a gradient optimization in the future, we must specify

x = torch.randn(3, requires_grad=True)

And given the function

y = x + 2

- Note when functions trail with _ (.mean_) it means that pytorch will apply
the operator in place

## Backpropagation

Goal is to minimize the gradient of your model using the chain rule

## Dataset and Dataloader

Some definitions for torch models

epoch = 1 forward and backward pass of ALL training samples

batch_size = number of training samples in one forward and backward pass

number of iterations = number of passes, each pass using [batch_size] num of samp 
