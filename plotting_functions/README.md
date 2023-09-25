In this folder a number of scripts are contained to create functions. 

If the plot are more complex (e.g. plot cross-section) the script only serves this purpose. In that case the purpose is clearly indicated by the name of the script (excpet plot_functions.py that seems to only plot cross-sections).

Other scripts are capable of plotting more than one type of plot. In the course of my thesis I found it most useful if the functions take the Axes object to which the boxplots are added  as input. This allows to treat these plots more flexible and adjust them to my needs later on.

In the case of trajectory evaluation, this modular plots (e.g. plot helpers) are always used in a more elaborate way. These scripts, (I try to add a "make_plot") in the name, output the finished figure. Since this is not very flexible, the figures might need adjusting before using them in the thesis.
