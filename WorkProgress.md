# Analysis of Face vs Places using Box plot:

### What is Box Plot?

I've learnt about boxplot from [this](https://plot.ly/chart-studio-help/what-is-a-box-plot/) site and created one using [matplotlib](https://matplotlib.org/3.1.3/api/_as_gen/matplotlib.pyplot.boxplot.html)

### Data Collection:

I've looked at all the 1200 training images and collected the image names of those images which have human faces in them. There were around 70 such images. Then I picked the categories which were majorly dominated by place aspect. This counted to 9 categories => 72 images in total. The dataset of Faces and Places being of almost same in number I've further continued with the experiment.

### Process:

Inorder to create a box-plot the pre-processed voxels of all the 10 rois of the PRE-GOD dataset were taken.
The voxels of certain ROI related to the place images were all meaned per subject and across all the subjects forming an array of 1X6 (5 subjects + avg of all subjects). This is done for all the ROIs. This data was then used in creating a box plot for Place related images. Similarly, the same process was follwed for the face related images and the box plot was plotted

### Observations:

- Clear distinction in the FFA and PPA is observed in the box plot for face and the place related images.
- The median of FFA is more than PPA in Face related images.
- The median of PPA is more than FFA in Place related images.
- Even though there is variation in the brain activation in FFA and PPA regions the value is not much significant.

# Hierarchical Clustering / Dendogram:

### What is done?

Dendogram is plottedusing the raw voxels for each of the subjects for all the 150 categories. The voxels within the category are averaged resulting in 150X(voxel_size) which is used for platting the dendogram.

### Observations:

- The number of clusters varied for each of the ROI.
- The number of clusters observed were in the range of 2 to 6.
- The higher regions of brain had higher number of clusters.

### Challenges:

- Average of the voxels for a single ROI could not be calculated as the voxel size per ROI is different for different subjects.
- What to do next?

# Plant vs Animal vs Inanimate:

### Data Collection:

- 3 super categories - Plants, Animals, Inanimate
- 8 categories in each of the super categories.
- Totaof 192 images.

### What was done?

- RDM construction of 192x192, 24x24, 3x3 for each subject and the average of all subjects were plotted
- Construction of Box Plot for these three super categories was done by using the raw voxels of all images which belonged to that super category.

### Observations:

- In the box plot, FFA has the maximum median in Animals, V1 in Plants and PPA in inanimate.
- Observing the RDMs it is found that for PPA => Plants and Inanimate are highly correlated and for V1, V2 Plants and animals are highly correlated. For FFA Animals is dark blue and the rest areconsidered different.
