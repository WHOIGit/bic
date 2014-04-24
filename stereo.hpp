#pragma once
#include <opencv2/opencv.hpp>

/**
 * Utilities for computing pixel offset between side-by-side stereo pairs.
 *
 * @author Joe Futrelle
 *
 * @brief stereo alignment for uncalibrated images
 */
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
   * @param if sample variance is needed (a metric of quality), a pointer
   * to a variable to use to return it
   *
   * @return the pixel offset between the images
   */
  int align(cv::Mat y_LR, int template_size=64, double* vary=NULL);
  /**
   * Given a stereo pair and an x offset, return the overlap region
   * of the left camera image.
   * @param LR a left/right side-by-side stereo image pair
   */
  cv::Rect overlap_L(cv::Mat LR, int xoff=0);
  /**
   * Given a stereo pair and an x offset, return the overlap region
   * of the right camera image.
   * @param LR a left/right side-by-side stereo image pair
   */
  cv::Rect overlap_R(cv::Mat LR, int xoff=0);
  /**
   * Given a stereo pair and x offset, generate a crosseye view
   * of just the overlapping region. If no offset is given,
   * just swap the sides of the image
   */
  cv::Mat xeye(cv::Mat LR, int xoff=0);
  /**
   * Given a stereo pair and x offset, generate a side-by-side
   * view at a specific output image size, and scale/pad (i.e.,
   * "letterbox") the two images to fit the output image size.
   * @param LR original side-by-side image
   * @param width width of output image
   * @param height height of output image
   * @param xoff x offset in original image coordinates
   */
  cv::Mat sideBySide(cv::Mat LR, int width, int height, int xoff=0);

  /**
   * The metrics associated with a stereo pair of cameras. Provides
   * conversions and utilities for computing where corresponding points
   * lie in image pairs.
   */
  class CameraPair {
  public:
    /** distance between cameras */
    double camera_sep;
    /** effective focal length */
    double focal_length;
    /** distance between pixel centers on sensor */
    double pixel_sep;
    /** adjustment factor for underwater images */
    double h2o_adjustment;
    CameraPair() { }
    /**
     * @param camera_sep distance between cameras
     * @param focal_length effective focal length
     * @param pixel_sep distance between pixel centers on sensor
     * @param h2o_adjustment scaling factor for underwater images (default 1.2)
     */
    CameraPair(double camera_sep, double focal_length, double pixel_sep, double h2o_adjustment=1.2) {
      this->camera_sep = camera_sep;
      this->focal_length = focal_length;
      this->pixel_sep = pixel_sep;
      this->h2o_adjustment = h2o_adjustment;
    }
    /**
     * Convert a pixel offset along the x axis between a stereo pair to altitude
     * @param xoff the x offset
     * @return the altitude
     */
    double xoff2alt(int xoff) {
      return (camera_sep * focal_length * h2o_adjustment) / (xoff * pixel_sep);
    }
    /**
     * Convert an altitude to a pixel offset between a stereo pair.
     * @param alt the altitude
     * @return pixel offset (not rounded to nearest pixel)
     */
    double alt2xoff(double alt) {
      return (camera_sep * focal_length * h2o_adjustment) / (alt * pixel_sep);
    }
  };
  //
  void xoff_test(int argc, char **argv); // FIXME testing
}
