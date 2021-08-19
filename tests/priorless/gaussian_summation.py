import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage

def switch_off(img, rate):
    img_flat = img.flatten()
    n = len(img_flat)
    off_idc = np.random.choice(np.arange(n), size=int(round(n*rate)), replace=False)
    img_flat[off_idc] = 0
    return img_flat.reshape(*img.shape)


shape = (1000, 1000)
n = np.prod(shape)
amplitude = 1
switch_off_rate = 0.9


# img = np.random.rand(*shape)
# img = np.zeros((shape))
# img[500, 500] = 1

n_combinations = 1
imgs = []
for i in range(n_combinations):
    switch_off_rate = np.random.rand()

    smooth_x = np.random.uniform(1, 100)
    smooth_y = np.random.uniform(1, 100)

    new_img = np.random.rand(*shape)
    new_img = switch_off(new_img, switch_off_rate)
    
    new_img = ndimage.gaussian_filter(new_img, sigma=(smooth_x, smooth_y), order=0)

    imgs.append( new_img )

# Scale images
scalers = [np.random.rand() for _ in range(n_combinations)]
imgs = [img*scaler for img, scaler in zip(imgs, scalers)]
img = np.mean(imgs, axis=0)

plt.figure()
plt.imshow(img, cmap='gray', vmin=img.min(), vmax=img.max())
plt.colorbar()
plt.show()


