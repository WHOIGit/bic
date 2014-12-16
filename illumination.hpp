#pragma once
#include <exception>
#include <stdexcept>
#include <string>
#include <fstream>
#include <boost/thread.hpp>
#include <boost/format.hpp>
#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/predicate.hpp>

#include "interpolation.hpp"
#include "stereo.hpp"

namespace fs = boost::filesystem;

#define R_PREFIX "R_"
#define G_PREFIX "G_"
#define B_PREFIX "B_"

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
 *
 * @author Joe Futrelle
 *
 * @brief frame averaging for illumination correction
 */
namespace illum {
  class Lightfield;
  class MultiLightfield;
  class RgbLightfield;
  void correct(cv::InputArray src, cv::OutputArray dst, cv::Mat average);
};

/**
 * Lightfield provides frame-averaging for any sequence of
 * single-channel images. It also allows for input images to partially
 * contribute to the lightfield on either a per-image or per-pixel
 * basis, to allow for more complex applications (see
 * MultiLightfield).
 *
 * All images must be the same dimensions. No interpolation,
 * processing, scaling, or adjustment is done to images or pixel
 * values. The expected dimensions are determined by the first image
 * added.
 *
 * Initialization is lazy; one simply constructs this object, adds
 * images, and then requests the average. The average can be requested
 * at any time and images can be added after it that contribute to a
 * new average.
 *
 * A lightfield can be saved and loaded from a 16-bit TIFF
 * image. Lightfieldss that have been loaded can have new images added to
 * them and then the lightfields can be saved again.
 */
class illum::Lightfield {
  typedef cv::Mat Mat;
  typedef std::string string; 
  string loadpath; // pathname to load from (in deferred mode)
  Mat sum; // running sum
  Mat count; // running count (fractional)
  void init(Mat image) {
    if(image.empty())
      throw std::runtime_error("cannot initialize lightmap with empty image");
    if(empty()) {
      int h = image.size().height;
      int w = image.size().width;
      sum = Mat::zeros(h, w, CV_32F);
      count = Mat::zeros(h, w, CV_32F);
    }
  }
  void validate(Mat image) {
    init(image);
    if(image.size() != sum.size())
      throw std::runtime_error("input image not the same size as lightfield");
  }
public:
  /**
   * Construct an empty lightfield, ready for images to be added to
   * it.
   */
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
   *   image to the average per-pixel (range 0-1)
   */
  void addImage(Mat image, Mat alpha) {
    validate(image);
    if(image.size() != alpha.size())
      throw std::runtime_error("alpha channel not the same size as image");
    Mat image32f;
    image.convertTo(image32f, CV_32F); // convert to floating point
    sum += image32f.mul(alpha); // multiply by alpha and add to sum image
    count += alpha; // add alpha channel to count image
  }
  /**
   * Add an image to the running total.
   *
   * @param image the image to add
   * @param alpha an alpha value specifying the contribution of the image to
   * the average (default 1.0)
   */
  void addImage(Mat image, double alpha=1.0) {
    validate(image);
    Mat image32f;
    image.convertTo(image32f, CV_32F); // convert to floatin point
    if(alpha != 1.0) {
      sum += image32f * alpha; // multiply by alpha and add to sum image
    } else {
      sum += image32f;
    }
    count += alpha; // add alpha to count image
  }
  /**
   * Compute the average image and return it in dst.
   * dst must be of type CV_32F.
   *
   * @param _dst the destination image
   */
  void getAverage(cv::OutputArray _dst) {
    Mat avg = getAverage();
    _dst.create(avg.size(), avg.type());
    Mat dst = _dst.getMat();
    if(dst.type() != CV_32F)
      throw std::runtime_error("output image must be 32-bit floating point");
    avg.copyTo(dst);
  }
  /**
   * Compute the average image and return it. The returned image will be
   * of type CV_32F.
   *
   * @return a new image containing the average of all the images
   * added
   */
  Mat getAverage() {
    if(empty() && !loadpath.empty())
      load(loadpath);
    if(empty())
      throw std::runtime_error("average requested before any images were added");
    // average image is just sum image divided by count image
    return sum / count;
  }
  /**
   * Return the maximum and minimum image counts for this lightmap.
   */
  void minMaxCount(double* minCount, double* maxCount) {
    cv::minMaxLoc(count, minCount, maxCount);
  }
  /**
   * Save the lightfield to an image file. The image file will contain
   * both average and count information, so will be twice the
   * dimensions of the input images.
   *
   * @param pathname the file to save to. Must end with a TIFF extension
   * e.g., ".tiff"
   * @param ensure_unity adjust lightmap counts so that they are >= 1
   * to prevent zeros in the count image
   */
  void save(string pathname, bool ensure_unity=false) {
    using cv::Rect;
    using boost::algorithm::ends_with;
    string lp = pathname;
    boost::to_lower(lp);
    if(!(ends_with(lp,".tif") || ends_with(lp,".tiff")))
      throw std::runtime_error("lightfield output pathname does not end with .tif or .tiff");
    Mat adj_sum = sum;
    Mat adj_count = count;
    if(ensure_unity) {
      double minCount, maxCount;
      minMaxCount(&minCount, &maxCount);
      if(minCount < 1) {
	double scale = 1.0 / minCount;
	adj_sum *= scale;
	adj_count *= scale;
      }
    }
    Mat average = adj_sum / adj_count; // FIXME tile from getAverage
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
    if(!composite_16u.data)
      throw std::runtime_error("unable to read lightmap image file");
    if(composite_16u.type() != CV_16U)
      throw std::runtime_error("lightfield image file must be 16-bit unsigned");
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
  /**
   * Defer loading this lightmap until the average is requested
   * @param pathname the pathname to load from
   */
  void load_later(string pathname) {
    loadpath = pathname;
  }
};

/**
 * One slice of a multi-lightfield. This internal class should not
 * be used elsewhere.
 */
template <typename T> class Slice {
private:
  boost::mutex mutex; // for threadsafety
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
  boost::mutex* get_mutex() {
    return &mutex;
  }
};

/**
 * A multi-altitude lightfield consisting of a set of lightfields each
 * of which is associated with a specific altitude class. Altitude classes
 * are evenly-spaced, starting from 0, according to a parameter passed into
 * the constructor.
 *
 * Average images are generated from arbitrary altitudes in the
 * multi-lightfield's altitude range via interpolation.
 *
 * Interpolation accepts pitch and roll parameters in addition to altitude,
 * which are used to estimate per-pixel distance to the substrate based on
 * a simple geometric model of the relative configuration of the camera and
 * an assumed flat substrate.
 */
class illum::MultiLightfield {
protected:
  boost::mutex* slices_mutex;
  typedef cv::Mat Mat;
  typedef std::string string; 
  typename std::vector<Slice<int>* > slices; // altitude slices
  double alt_step;
  int undertrain;
  int overtrain;
  Slice<int>* getSlice(int i) { // accessor for slice by altitude bin
    boost::lock_guard<boost::mutex> lock(*slices_mutex); // protect entire method with mutex
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
   * @param undertrain undertraining threshold (in number of images)
   * @param overtrain overtraining threshold (in number of images)
   */
  MultiLightfield(double step_m=0.1, int undertrain=20, int overtrain=65535) {
    slices_mutex = new boost::mutex();
    alt_step = step_m;
    this->undertrain = undertrain;
    this->overtrain = overtrain;
  }
  /**
   * Add an image to the lightfield
   * @param image the image to add
   * @param alt the altitude the image was taken at
   * @return number of lightfield slices affected (0-2)
   */
  int addImage(Mat image, double alt) {
    int n_slices = 0;
    int i = alt / alt_step;
    int j = i + 1;
    double Wi = ((alt_step * j) - alt) / alt_step;
    double Wj = (alt - (alt_step * i)) / alt_step;
    Slice<int>* slice = getSlice(i);
    boost::mutex* mutex = slice->get_mutex();
    double minCount, maxCount;
    { // protect slice with mutex to prevent concurrent writes
      boost::lock_guard<boost::mutex> lock(*mutex);
      slice->getLightfield()->minMaxCount(&minCount, &maxCount);
      if(minCount < overtrain) { // skip image if overtrained
	slice->getLightfield()->addImage(image, Wi);
	n_slices++;
      }
    }
    slice = getSlice(j);
    mutex = slice->get_mutex();
    { // protect slice with mutex to prevent concurrent writes
      boost::lock_guard<boost::mutex> lock(*mutex);
      slice->getLightfield()->minMaxCount(&minCount, &maxCount);
      if(minCount < overtrain) { // skip image if overtrained
	slice->getLightfield()->addImage(image, Wj);
	n_slices++;
      }
    }
    return n_slices;
  }
  /**
   * Get the average image at the given altitude.
   * If the altitude is not located exactly at one of the altitude bins,
   * the average image is interpolated between any overlapping bins.
   * @param _dst the output image (zeros at desired resolution)
   * @param alt the altitude the image was taken at
   */
  void getAverage(cv::OutputArray _dst, double alt) {
    using boost::format;
    using boost::str;
    int i = alt / alt_step;
    int j = i + 1;
    double Wi = ((alt_step * j) - alt) / alt_step;
    double Wj = (alt - (alt_step * i)) / alt_step;
    Mat Ai;
    double minCountI, maxCountI, countI;
    Slice<int>* slice = getSlice(i);
    boost::mutex* mutex = slice->get_mutex();
    { // protect slice with mutex to prevent concurrent writes
      boost::lock_guard<boost::mutex> lock(*mutex);
      Ai = slice->getLightfield()->getAverage();
      slice->getLightfield()->minMaxCount(&minCountI, &maxCountI);
    }
    Mat Aj;
    double minCountJ, maxCountJ, countJ;
    slice = getSlice(j);
    mutex = slice->get_mutex();
    { // protect slice with mutex to prevent concurrent writes
      boost::lock_guard<boost::mutex> lock(*mutex);
      Aj = slice->getLightfield()->getAverage();
      slice->getLightfield()->minMaxCount(&minCountJ, &maxCountJ);
    }
    // determine training level of each image
    countI = minCountI; // FIXME is this the right metric for distance-mapping case?
    countJ = minCountJ; // FIXME is this the right metric for distance-mapping case?
    if(countI < undertrain && countJ < undertrain) {
      throw std::runtime_error(str(format("lightmap is undertrained at %.2f") % alt));
    } else if(countI < undertrain) { // I is undertrained
      if(Wi > Wj) { // and we're closest to I
	throw std::runtime_error(str(format("lightmap is undertrained at %.2f") % alt));
      } else { // or we're closer to J
	Wi = 0; Wj = 1; // do not use slice I and just use slice J
      }
    } else if(countJ < undertrain) { // J is undertrained
      if(Wj > Wi) { // and we're closest to J
	throw std::runtime_error(str(format("lightmap is undertrained at %.2f") % alt));
      } else { // or we're closest to I
	Wi = 1; Wj = 0; // do not use slice J and just use slice I
      }
    }
    // ensure destination image is not empty
    _dst.create(Ai.size(), Ai.type());
    Mat dst = _dst.getMat();
    if(dst.type() != CV_32F)
      throw std::runtime_error("output image must be 32-bit floating point");
    // now set it to the average
    dst = (Ai * Wi) + (Aj * Wj);
    return;
  }
  /**
   * Save the multi-lightfield to a directory. The lightfield is
   * saved as a set of TIFF images. If the directory contains files
   * with similar names, this will overwrite those files, so don't
   * use a directory that already has other files in it.
   *
   * @param outdir the output directory
   * @param prefix optional prefix for slice files
   */
  void save(string outdir, string prefix = "") {
    typename std::vector<Slice<int>* >::iterator it = slices.begin();
    for(; it != slices.end(); ++it) {
      Slice<int>* slice = *it;
      illum::Lightfield* lf = slice->getLightfield();
      if(!lf->empty()) {
	int count = slice->getAlt();
	std::stringstream outpaths;
	outpaths << prefix << "slice_" << count << ".tiff";
	fs::path p(outdir);
	p /= outpaths.str();
	lf->save(p.string());
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
   * @param prefix optional prefix on filenames
   * @return number of slices loaded
   */
  int load(string outdir, string prefix = "") {
    int loaded = 0;
    for(int count = 0; count < 1000; ++count) {
      std::stringstream inpaths;
      inpaths << prefix << "slice_" << count << ".tiff";
      fs::path p(outdir);
      p /= inpaths.str();
      if(fs::exists(p)) {
	Slice<int>* slice = getSlice(count);
	slice->getLightfield()->load_later(p.string()); // defer load
	loaded++;
      }
    }
    return loaded;
  }
};

class illum::RgbLightfield {
  typedef cv::Mat Mat;
  typedef std::string string; 
protected:
  illum::MultiLightfield* R;
  illum::MultiLightfield* G;
  illum::MultiLightfield* B;
public:
  /**
   * Create a multi-altitude lightfield.
   * @param step_m the width of each altitude bin in m (default: 10cm)
   * @param undertrain undertraining threshold (in number of images)
   * @param overtrain overtraining threshold (in number of images)
   */
  RgbLightfield(double step_m=0.1, int undertrain=20, int overtrain=65535) {
    R = new MultiLightfield(step_m, undertrain, overtrain);
    G = new MultiLightfield(step_m, undertrain, overtrain);
    B = new MultiLightfield(step_m, undertrain, overtrain);
  }
  /**
   * Add an image to the lightfield
   * @param image the image to add
   * @param alt the altitude the image was taken at
   * @return number of lightfield slices affected (0-2)
   */
  int addImage(Mat image, double alt) {
    cv::Mat bgr_image;
    // extract color channels
    std::vector<cv::Mat> channels;
    cv::split(bgr_image, channels);
    B->addImage(channels[0], alt);
    G->addImage(channels[1], alt);
    R->addImage(channels[2], alt);
  }
  /**
   * Get the average image at the given altitude.
   * If the altitude is not located exactly at one of the altitude bins,
   * the average image is interpolated between any overlapping bins.
   * @param _dst the output image (zeros at desired resolution)
   * @param alt the altitude the image was taken at
   */
  void getAverage(cv::OutputArray _dst, double alt) {
    cv::Mat Ra, Ga, Ba;
    R->getAverage(Ra, alt);
    B->getAverage(Ga, alt);
    G->getAverage(Ba, alt);

    // create BGR color image from each channel average
    std::vector<cv::Mat> bgr;
    bgr.push_back(Ba);
    bgr.push_back(Ga);
    bgr.push_back(Ra);
    cv::Mat dst = _dst.getMat();
    if(dst.type() != CV_32F)
      throw std::runtime_error("output image must be 32-bit floating point");
    cv::merge(bgr, dst);
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
    R->save(outdir, R_PREFIX);
    G->save(outdir, G_PREFIX);
    B->save(outdir, B_PREFIX);
  }
  /**
   * Load the multi-lightfield from a directory. The lightfield is
   * saved as a set of TIFF images. If the directory contains files
   * with similar names, this will attempt to read them, so don't use
   * a directory that wasn't populated using the save method.
   *
   * @param outdir the output directory where the lightfield is stored
   * @return number of slices loaded
   */
  int load(string outdir) {
    R->load(outdir, R_PREFIX);
    G->load(outdir, G_PREFIX);
    B->load(outdir, B_PREFIX);
  }
};
