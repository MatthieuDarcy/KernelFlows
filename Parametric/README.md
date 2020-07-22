Here is a brief overview of the differences between the autograd and Frechet versions of Parametric Kernel Flows.

Autograd version:

Requirements: autograd (https://github.com/HIPS/autograd).

Pros:
  Implementing a new kernel is (generally) easy as long as the implementation does not break autograd rules.
  Implementing a new kernel does not involve specifying the derivative of the kernel.

Cons:
  Slighlty slower than the Frechet version.
 
 Frechet version:
 
 Requirements: numpy.
 
 Pros:
  Slighlty faster than the autograd version
  Implementing a new kernel does not require any knowledge of autograd's rules
  
  Cons:
    Implementing a new kernel requires implementing its derivative.
    
 
 If you are not planning on implementing new kernels, it is recommended to use the Frechet version as it is faster.

  
