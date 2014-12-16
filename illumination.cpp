#include "illumination.hpp"

/**
 * Correct an image by dividing it by an illumination model image,
 * e.g., one computed by frame averaging. The corrected image, whose
 * values in floating-point representation are >1 for every pixel in
 * the input image that is brighter than the corresponding pixel in
 * the lightmap, is returned. Therefore the output array must be
 * floating-point.
 *
 * @param _src the source image
 * @param _dst the destination image (must be same size as src and CV_32F type)
 * @param average the appropriate average image returned by the lightfield
 * @return a new, correctly-illuminated image (in the output array's type)
 */
void illum::correct(cv::InputArray _src, cv::OutputArray _dst, cv::Mat average) {
  using namespace cv;
  Mat src = _src.getMat();
  _dst.create(src.size(), CV_32F);
  Mat dst = _dst.getMat();
  if(src.size() != dst.size())
    throw std::runtime_error("input/output image size mismatch");
  if(dst.type() != CV_32F)
    throw std::runtime_error("output image must be floating-point type");
  if(src.size() != average.size())
    throw std::runtime_error("input image not same size as lightfield");
  Mat src32f;
  src.convertTo(src32f, CV_32F); // convert to floating point
  // divide image by lightfield
  Mat average32f;
  if(average.type() != CV_32F) {
    average.convertTo(average32f, CV_32F);
  } else {
    average32f = average;
  }
  // now compute corrected image in its floating-point representation
  Mat correct32f = src32f / average32f;
  correct32f.copyTo(dst);
}
