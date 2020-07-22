# KernelFlows

Here you can find my implementation of the Kernel Flows algorithm (for details see: Kernel Flows: from learning kernels from data into the abyss, Houman Owhadi, Gene Ryan Yoo at https://arxiv.org/abs/1808.04475).

There are two main versions of Kernel Flows: parametric and non-parametric. The non-parametric is not available as of yet.

The parametric version is implemented in two different ways: the first directly computes the Frechet derivative of the rho function, the second passes the rho function to autograd (see: https://github.com/HIPS/autograd) which computes its derivative. For advantages and disadvantages, see the readme file in the parametric folder.

Quick overview:

The basic idea behind kernel flows is that a good kernel is one which does not suffer significant loss when losing half of its training point.
This leads to the formulation of a loss function, refered to as rho, which is a measure of how good the kernel is with half the points compared to the full points. 
Kernel Flows then optimizes the parameters of the kernel through gradient descent on the rho function.
