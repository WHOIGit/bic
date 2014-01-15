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
 * Basic usage involves adding images, followed by correcting images.
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
  template <typename T> class MultiLightfield;
  void correct(cv::Mat src, cv::Mat dst, cv::Mat lightfield);
};

class illum::Lightfield {
  typedef cv::Mat Mat;
  typedef std::string string; 
  Mat sum; // running sum
  Mat count; // running count (fractional)
  // lazy initialization based on size of first image
  void init(Mat image) {
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
  cv::Mat getAverage() {
    return lf->getAverage();
  }
  T getAlt() {
    return a;
  }
};

template <typename T> class illum::MultiLightfield {
  typedef cv::Mat Mat;
  typedef std::string string; 
private:
  typename interp::LinearBinning<T> binning;
  typename std::vector<Slice<T>* > slices;
  Slice<T>* getSlice(T alt) {
    typename std::vector<Slice<T>* >::iterator it = slices.begin();
    for(; it != slices.end(); ++it) {
      Slice<T>* slice = *it;
      if(slice->getAlt() == alt) {
	return slice;
      }
    }
  }
public:
  MultiLightfield(T low, T high, T width) {
    binning = interp::LinearBinning<T>(low, high, width);
    std::vector<T> bins = binning.getBins();
    typename std::vector<T>::iterator it = bins.begin();
    for(; it != bins.end(); ++it) {
      slices.push_back(new Slice<T>(*it));
    }
  }
  void addImage(Mat image, T alt) {
    using namespace std;
    vector<pair<T,double> > result = binning.interpolate(alt);
    typename vector<pair<T,double> >::iterator it = result.begin();
    for(; it != result.end(); ++it) {
      pair<T,double> p = *it;
      T sAlt = p.first; // altitude
      double alpha = p.second; // contribution to this altitude's slice
      if(alpha > 0) {
	Slice<T>* slice = getSlice(sAlt);
	slice->getLightfield()->addImage(image, alpha);
      }
    }
  }
  Mat getAverage(T alt) {
    Mat average;
    std::vector<std::pair<T,double> > result = binning.interpolate(alt);
    typename std::vector<std::pair<T,double> >::iterator it = result.begin();
    for(; it != result.end(); ++it) {
      std::pair<T,double> p = *it;
      T sAlt = p.first; // altitude
      double alpha = p.second; // contribution to this altitude's slice
      if(alpha > 0) {
	// nonzero contribution. find the slice
	Slice<T>* slice = getSlice(sAlt);
	Mat sAverage = slice->getAverage();
	if(average.empty()) {
	  average.create(sAverage.size(), CV_32F);
	}
	average += sAverage.mul(alpha);
      }
    }
    // FIXME memoize or at least cache slice averages
    return average;
  }
  void save(string outdir) {
    typename std::vector<Slice<T>* >::iterator it = slices.begin();
    for(int count = 0; it != slices.end(); ++it, ++count) {
      Slice<T>* slice = *it;
      illum::Lightfield* lf = slice->getLightfield();
      if(!lf->empty()) {
	std::stringstream outpaths;
	outpaths << outdir << "/slice_" << count << ".tiff";
	lf->save(outpaths.str());
      }
    }
  }
  void load(string outdir) {
    std::vector<T> bins = binning.getBins();
    typename std::vector<Slice<T>* >::iterator sit = slices.begin();
    for(int count = 0; sit != slices.end(); ++sit, ++count) {
      std::stringstream inpaths;
      inpaths << outdir << "/slice_" << count << ".tiff";
      if(access(inpaths.str().c_str(),F_OK) != -1) {
	Slice<T>* slice = *sit;
	slice->getLightfield()->load(inpaths.str());
      }
    }
  }
};
