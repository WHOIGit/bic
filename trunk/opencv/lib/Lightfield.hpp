#pragma once
#include <exception>
#include <opencv2/opencv.hpp>

/**
 * Use Lightfield to accumulate a frame average from multiple images
 * that are the same size, and then "correct" images by dividing them
 * by the average image. This has the effect of flattening the
 * lightfield so that all of the corrected image is evenly
 * illuminated, assuming that all images added and being corrected
 * have been acquired under the same lighting conditions.
 *
 * Basic usage involves adding images, followed by correcting images.
 *
 * In some cases an image might contribute partially to the frame
 * average, with the proportion varying per-pixel. In that case an
 * alpha channel can be passed when adding an image.
 *
 * All computations are in 32-bit floating point but any input image
 * type is supported. All images added and corrected should be the
 * same type or results will not be correct.
 *
 * Checkpointing is supported via saving and loading to the filesystem
 * in a 16-bit representation, so that running averages can be
 * computed without having to perform all the averaging in a single
 * program execution. Under this usage there is a limit of 65,535
 * images that can be averaged before overflowing the checkpoint image
 * format.
 */
// FIXME does this work with color images?
class Lightfield {
  typedef cv::Mat Mat;
  typedef std::string string; 
  Mat sum; // running sum
  Mat count; // running count (fractional)
  // lazy initialization based on size of first image
  void init(Mat image) {
    if(sum.empty()) {
      int h = image.size().height;
      int w = image.size().width;
      sum = Mat::zeros(h, w, CV_32F);
      count = Mat::zeros(h, w, CV_32F);
    }
  }
  void validate(Mat image) {
    init(image);
    assert(image.size().height == sum.size().height);
    assert(image.size().width == sum.size().width);
  }
public:
  Lightfield() { }
  /**
   * Add an image to the running total.
   *
   * @param image the image to add
   * @param alpha an alpha channel specifying the contribution of the
   *   image to the average per-pixel
   */
  void addImage(Mat image, Mat alpha) {
    validate(image);
    assert(image.size().height == alpha.size().height);
    assert(image.size().width == alpha.size().width);
    Mat image32f;
    image.convertTo(image32f, CV_32F); // convert to floating point
    sum += image32f.mul(alpha); // multiply by alpha and add to sum image
    count += alpha; // add alpha channel to count image
  }
  /**
   * Add an image to the running total.
   *
   * @param image the image to add
   */
  void addImage(Mat image, double alpha=1.0) {
    validate(image);
    Mat image32f;
    image.convertTo(image32f, CV_32F); // convert to floatin point
    sum += image32f * alpha; // multiply by alpha and add to sum image
    count += alpha; // add alpha to count image
  }
  /**
   * Compute the average image and return it.
   *
   * @return a new image containing the average of all the images
   * added
   */
  Mat getAverage() {
    assert(!sum.empty());
    // average image is just sum image divided by count image
    return sum / count;
  }
  /**
   * Correct an image by dividing it by the average image.
   *
   * @param image the image
   * @return a new, correctly-illuminated image
   */
  Mat correct(Mat image) {
    Mat average = getAverage();
    assert(image.size().height == average.size().height);
    assert(image.size().width == average.size().width);
    Mat image32f;
    image.convertTo(image32f, CV_32F); // convert to floating point
    // find intensity range of average image
    double minLightmap, maxLightmap;
    cv::minMaxLoc(average, &minLightmap, &maxLightmap);
    // divide image by average, then normalize to intensity range of average
    return (image32f / average) * (maxLightmap - minLightmap);
  }
  /**
   * Save the lightfield to an image file. The image file will contain
   * both average and count information, so will be twice the
   * dimensions of the input images.
   *
   * @param pathname the file to save to. Must end with a TIFF extension
   * e.g., ".tiff"
   */
  void save(string pathname) {
    using cv::Rect;
    // FIXME assert that the path ends ".tif" or equiv
    Mat average = getAverage();
    // create a composite image twice the height of the frame
    Mat composite;
    int h = average.size().height;
    int w = average.size().width;
    composite.create(h * 2, w, CV_32F);
    Mat top(composite, Rect(0, 0, w, h));
    Mat bottom(composite, Rect(0, h, w, h));
    average.copyTo(top); // average image goes in the top half
    count.copyTo(bottom); // count image goes in the bottom half
    // convert to 16-bit unsigned and save
    Mat composite_16u;
    composite.convertTo(composite_16u, CV_16U);
    cv::imwrite(pathname, composite_16u);
  }
  /**
   * Load the lightfield from an image file it was saved to.
   *
   * @param pathname the file that the lighfield was saved to
   */
  void load(string pathname) {
    using cv::Rect;
    // load the 16-bit unsigned checkpoint image
    Mat composite_16u = cv::imread(pathname, CV_LOAD_IMAGE_ANYDEPTH);
    assert(composite_16u.type() == CV_16U);
    int h = composite_16u.size().height / 2;
    int w = composite_16u.size().width;
    // convert to floating point
    Mat composite;
    composite_16u.convertTo(composite, CV_32F);
    Mat top(composite, Rect(0, 0, w, h));
    Mat bottom(composite, Rect(0, h, w, h));
    // initialize to the right frame size
    init(top);
    // now average and sum data into accumulation matrices
    top.copyTo(sum);
    bottom.copyTo(count);
    // and renormalize the average image to the count
    sum = sum.mul(count);
  }
};
