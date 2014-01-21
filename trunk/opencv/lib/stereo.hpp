#pragma once
#include <opencv2/opencv.hpp>

namespace stereo {
  /**
   * Estimate the alignment between a left-right stereo pair. This
   * algorithm does not require that the images be calibrated, but it
   * works poorly in extreme cases, such as very noisy images, images
   * with very large amounts of depth variation on small scales, images
   * that overlap only at the very edge (i.e., images where the dominant
   * objects are extremely close to the camera pair), or pairs of images
   * that were not taken at the same time.
   *
   * @param y_LR a single-channel image whose left half is the left
   * camera image and whose right half is the right camera image.
   * @param template_size the size of the image patch used to find
   * correspondences. Should be large enough to include image features
   * likely to be found in both images.
   *
   * @return the pixel offset between the images
   */
  int align(cv::Mat y_LR, int template_size=64);
}

void xoff_test(int argc, char **argv);
