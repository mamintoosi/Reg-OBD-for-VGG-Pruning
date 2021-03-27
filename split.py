import splitfolders  # or import split_folders

# Split with a ratio.
# To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
# splitfolders.ratio("input_folder", output="output", seed=1337, ratio=(.8, .1, .1), group_prefix=None) # default values

# Split val/test with a fixed number of items e.g. 100 for each set.
# To only split into training and validation set, use a single number to `fixed`, i.e., `10`.
splitfolders.fixed("C:/temp/Covid19/COVID-19_Radiography_Dataset", \
     output="C:/temp/git/OBD-for-VGG-Pruning-COVID19/data/COVID-Radiography", \
         seed=1337, fixed=(1000, 100, 100), oversample=False, group_prefix=None) # default values