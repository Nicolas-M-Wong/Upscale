# Upscale with class

The UPSCALE_CLASS.py is the main code, it contains all the necesserary code to be executed. Be aware that compare to the FUNCTION varaint you will loose certain fonctionnality. Must of them are : automatic renaming scheme (which means that it will be always named test.jpg), storing the performance in a text document.

The standard name of the upscaled photo is __fraise4.jpg__ . It can be renamed inside the python code of the file name UPSCALE_CLASS.py. It is located at the end of the program. The class take the name of the photo as argument.

*On my laptop this program upscale 256x256 picture to a 512x512 in 30 sec approximately (with a Ryzen 7 4700U).*

This version of the program is fully usable but lacks some of the features that you can found in the function/libraries version of the code. The main features that are lacking are the absence of wavelet/CSR compression, sharpness improvement.

The program might get stuck at 99% on one of the matrix calculus. I couldn't find the reason as it occures very rarely (I've only seen this behaviour twice). If it happens to you just need to do a keyboard interrupt and restart the program. The next execution should be fine.
