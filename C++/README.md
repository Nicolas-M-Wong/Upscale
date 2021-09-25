# Upscale
This programm is very far from release state. It is simply exploring the possibility of using C++ efficiency (even multithreading may be) to improve the usability of the initial program.
Without multithreading I can hope an efficiency improvement between x5 and x7 (the programm is executed inside a virtual machine) on my computer.

As of right now there is only the inverse matrix, determinant, transpose and matricial product developp using very basic array to handle the matrix. The objective is to be using the vector instead of array to make the programm more versatile.

The roadmap is the following for the pre release state:
 - *V0.1 get the basic mathematical function fonctionning with array*
 - *V0.2 add the different matrix and algorithm to have a functionnal resizing matrix program*

*Update 25-09-2021*
The V0.1 which includes the basic mathematical function is working. Now working on the V0.2 to applied it on the upscale program that I am trying to solve here. As of right now I am trying to understand how to use the Mat class used by openCV
