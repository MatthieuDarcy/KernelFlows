# KernelFlows

Here you can find my implementation of the Kernel Flows algorithm (for details see: Kernel Flows: from learning kernels from data into the abyss, Houman Owhadi, Gene Ryan Yoo at https://arxiv.org/abs/1808.04475).

There are two main versions of Kernel Flows: parametric and non-parametric. 

The parametric version is implemented in two different ways: the first directly computes the Frechet derivative of the rho function, the second passes the rho function to autograd (see: https://github.com/HIPS/autograd) which computes its derivative. For advantages and disadvantages, see the readme file in the parametric folder.

