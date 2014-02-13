
#pragma once
#include <exception>
#include <fstream>
#include <opencv2/opencv.hpp>

#include "interpolation.hpp"

/**
 * Use Lightfield to accumulate a frame average from multiple images
 * that are the same size, and then "correct" images by dividing them
 * by the average image. This has the effect of flattening the
 * lightfield so that all of the corrected image is evenly
 * illuminated, assuming that all images added and being corrected
 * have been acquired under the same lighting conditions.
 *
 * Basic usage involves adding images, followed by retrieving average
 * image which can be used to correct images via illum::correct.
 *
 * In some cases an image might contribute partially to the frame
 * average, with the proportion varying per-pixel. In that case an
 * alpha channel can be passed when adding an image.
 *
 * All computations are in 32-bit floating point but any
 * single-channel input image type is supported. All images added and
 * corrected should be the same type or results will not be correct.
 *
 * Checkpointing is supported via saving and loading to the filesystem
 * in a 16-bit representation, so that running averages can be
 * computed without having to perform all the averaging in a single
 * program execution. Under this usage there is a limit of 65,535
 * images that can be averaged before overflowing the checkpoint image
 * format.
 */
namespace illum {
  class Lightfield;
  class MultiLightfield;
  void correct(cv::InputArray src, cv::OutputArray dst, cv::Mat lightfield);
};

class illum::Lightfield {
  typedef cv::Mat Mat;
  typedef std::string string; 
  Mat sum; // running sum
  Mat count; // running count (fractional)
  // lazy initialization based on size of first image
  void init(Mat image) {
    assert(!image.empty());
    if(empty()) {
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
   * Is this lightfield empty or uninitialized?
   */
  bool empty() {
    return sum.empty();
  }
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
   * Compute the average image and return it in dst.
   *
   * dst must be of type CV_32F.
   */
  void getAverage(cv::OutputArray _dst) {
    Mat avg = getAverage();
    _dst.create(avg.size(), avg.type());
    Mat dst = _dst.getMat();
    assert(dst.type()==CV_32F);
    getAverage().copyTo(dst);
  }
  /**
   * Compute the average image and return it.
   *
   * @return a new image containing the average of all the images
   * added
   */
  Mat getAverage() {
    assert(!empty());
    // average image is just sum image divided by count image
    return sum / count;
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

// type parameter is the numerical type used for representing altitude
template <typename T> class Slice {
private:
  illum::Lightfield* lf; // model for this altitude
  T a; // altitude
public:
  Slice(T alt) {
    a = alt;
    lf = new illum::Lightfield();
  }
  illum::Lightfield* getLightfield() {
    return lf;
  }
  T getAlt() {
    return a;
  }
};

/**
 * A multi-altitude lightfield consisting of a set of lightfields each
 * of which is associated with a specific altitude. Average images are
 * generated from arbitrary altitudes in the multi-lightfield's
 * altitude range via interpolation.
 */
class illum::MultiLightfield {
  typedef cv::Mat Mat;
  typedef std::string string; 
private:
  typename std::vector<Slice<int>* > slices; // altitude slices
  double alt_step;
  double pixel_sep;
  double focal_length;
  Slice<int>* getSlice(int i) { // accessor for slice by altitude bin
    typename std::vector<Slice<int>* >::iterator it = slices.begin();
    for(; it != slices.end(); ++it) {
      Slice<int>* slice = *it;
      if(slice->getAlt() == i) {
	return slice;
      }
    }
    // the slice doesn't exist, create it
    Slice<int>* slice = new Slice<int>(i);
    slices.push_back(slice);
    return slice;
  }
public:
  /**
   * Create a multi-altitude lightfield.
   * @param step_m the width of each altitude bin in m (default: 10cm)
   * @param focal_length_m the effective focal length in m (default: 12mm)
   * @param pixel_sep_m the physical size of a pixel in m (default: 6.5um)
   */
  MultiLightfield(double step_m=0.1, double focal_length_m=0.012, double pixel_sep_m=0.0000065) {
    alt_step = step_m;
    focal_length = focal_length_m;
    pixel_sep = pixel_sep_m;
  }
  /**
   * Add an image to the lightfield
   * @param image the image to add
   * @param alt the altitude the image was taken at
   * @param pitch the pitch of the vehicle
   * @param roll the roll of the vehicle
   */
  void addImage(Mat image, double alt, double pitch, double roll) {
    // FIXME compute a low resolution and upscale
    // compute distance map
    int width = image.size().width;
    int height = image.size().height;
    Mat D = Mat::zeros(height, width, CV_32F);
    double width_m = width * pixel_sep;
    double height_m = height * pixel_sep;
    interp::distance_map(D, alt, pitch, roll, width_m, height_m, focal_length);
    // now discretize into slices
    int i = 0;
    Mat W = Mat::zeros(D.size(), CV_32F);
    while(cv::countNonZero(W) == 0) {
      interp::dist_weight(D, W, alt_step, i++);
    }
    while(cv::countNonZero(W) > 0) {
      Slice<int>* slice = getSlice(i);
      slice->getLightfield()->addImage(image, W);
      interp::dist_weight(D, W, alt_step, i++);
    }
  }
  /**
   * Get the average image at the given altitude.
   * If the altitude is not located exactly at one of the altitude bins,
   * the average image is interpolated between any overlapping bins.
   * @param dst the output image (zeros at desired resolution)
   * @param alt the altitude the image was taken at
   * @param pitch the pitch of the vehicle
   * @param roll the roll of the vehicle
   */
  void getAverage(cv::OutputArray _dst, double alt, double pitch, double roll) {
    Mat dst = _dst.getMat();
    // FIXME tiles
    int width = dst.size().width;
    int height = dst.size().height;
    Mat D = Mat::zeros(height, width, CV_32F);
    double width_m = width * pixel_sep;
    double height_m = height * pixel_sep;
    interp::distance_map(D, alt, pitch, roll, width_m, height_m, focal_length);
    // now discretize into slices
    int i = 0;
    Mat W = Mat::zeros(D.size(), CV_32F);
    while(cv::countNonZero(W) == 0) {
      interp::dist_weight(D, W, alt_step, i++);
    }
    while(cv::countNonZero(W) > 0) {
      Slice<int>* slice = getSlice(i);
      Mat sAverage = slice->getLightfield()->getAverage();
      dst += sAverage.mul(W); // multiply by slice weight
      interp::dist_weight(D, W, alt_step, i++);
    }
  }
  /**
   * Save the multi-lightfield to a directory. The lightfield is
   * saved as a set of TIFF images. If the directory contains files
   * with similar names, this will overwrite those files, so don't
   * use a directory that already has other files in it.
   *
   * @param outdir the output directory
   */
  void save(string outdir) {
    typename std::vector<Slice<int>* >::iterator it = slices.begin();
    for(; it != slices.end(); ++it) {
      Slice<int>* slice = *it;
      illum::Lightfield* lf = slice->getLightfield();
      if(!lf->empty()) {
	int count = slice->getAlt();
	std::stringstream outpaths;
	outpaths << outdir << "/slice_" << count << ".tiff";
	lf->save(outpaths.str());

      }
    }
  }
  /**
   * Load the multi-lightfield from a directory. The lightfield is
   * saved as a set of TIFF images. If the directory contains files
   * with similar names, this will attempt to read them, so don't use
   * a directory that wasn't populated using the save method.
   *
   * @param outdir the output directory where the lightfield is stored
   */
  void load(string outdir) {
    for(int count = 0; count < 1000; ++count) {
      std::stringstream inpaths;
      inpaths << outdir << "/slice_" << count << ".tiff";
      if(access(inpaths.str().c_str(),F_OK) != -1) {
	Slice<int>* slice = getSlice(count);
	slice->getLightfield()->load(inpaths.str());
      }
    }
  }
};
