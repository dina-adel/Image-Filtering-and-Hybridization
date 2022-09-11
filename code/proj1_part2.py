
from helpers import vis_hybrid_image, load_image, save_image, gen_hybrid_image

# Read images and convert to floating point format
image1 = load_image('../data/dog.bmp')
image2 = load_image('../data/cat.bmp')

# image1 = load_image('../data/einstein.bmp')
# image2 = load_image('../data/marilyn.bmp')

# Hybrid Dog & Cat Images
cutoff_frequency = 6
size = 23
# marilyn & albert
# cutoff_frequency = 9
# size = 9

# Merging the two images
low_frequencies, high_frequencies, hybrid_image = gen_hybrid_image(image1, image2, cutoff_frequency, size, 'zero')

# Visualize and save outputs ##
vis = vis_hybrid_image(hybrid_image)
save_image('../results/low_frequencies.jpg', low_frequencies)
save_image('../results/high_frequencies.jpg', high_frequencies)
save_image('../results/hybrid_image.jpg', hybrid_image)
save_image('../results/hybrid_image_scales.jpg', vis)
