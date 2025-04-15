
import cv2
import numpy as np

"""
Implementation of the XDoG algorithm for Line Extraction
"""

def DoG(image, size, sigma, k=1.6, gamma=1.):
    g1 = cv2.GaussianBlur(image, (size, size), sigma)
    g2 = cv2.GaussianBlur(image, (size, size), sigma*k)
    return g1 - gamma * g2

def XDoG(image, size, sigma, eps, phi, k=1.6, gamma=1.):
    eps /= 255
    d = DoG(image, size, sigma, k, gamma)
    d /= d.max()
    e = 1 + np.tanh(phi * (d - eps))
    e[e >= 1] = 1
    return e * 255

# This config is found by the author
# modify if not the desired output
XDoG_config = dict(
    size=0,
    sigma=0.5,
    eps=0.5,
    phi=100,
    k=3,
    gamma=0.97
)

def gen_xdog_image(src, dst):
    gray = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
    # I wanted the gamma between [0.97, 0.98]
    # but it depends on the image so I made it move randomly
    # comment out if this is not needed
    # XDoG_config['gamma'] += 0.01 * np.random.rand(1)
    dogged = XDoG(gray, **XDoG_config)
    cv2.imwrite(dst, dogged)

if __name__ == "__main__":
    gen_xdog_image('haruhi_reference.png', 'result/haruhi.png')