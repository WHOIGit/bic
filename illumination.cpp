#include "illumination.hpp"

/**
 * Correct an image by dividing it by an illumination model image,
 * e.g., one computed by frame averaging. The corrected image, whose
 * values in floating-point representation are >1 for every pixel in
 * the input image that is brighter than the corresponding pixel in
 * the lightmap, are scaled by 0.33 of the output array's data
 * type. To avoid clipping and perform your own adjustment, use a
 * floating-point result type. Supported types are CV_8U, CV_16U,
 * and CV_32F.
 *
 * @param _src the source image
 * @param _dst the destination image (must be same size and type as src)
 * @param lightfield the appropriate average image returned by the lightfield
 * @return a new, correctly-illuminated image (in the output array's type)
 */
void illum::correct(cv::InputArray _src, cv::OutputArray _dst, cv::Mat lightfield) {
  using namespace cv;
  Mat src = _src.getMat();
  _dst.create(src.size(), src.type());
  Mat dst = _dst.getMat();
  if(src.size() != dst.size())
    throw std::runtime_error("input/output image size mismatch");
  if(src.size() != lightfield.size())
    throw std::runtime_error("input image not same size as lightfield");
  Mat src32f;
  src.convertTo(src32f, CV_32F); // convert to floating point
  // divide image by lightfield
  Mat lightfield32f;
  if(lightfield.type() != CV_32F) {
    lightfield.convertTo(lightfield32f, CV_32F);
  } else {
    lightfield32f = lightfield;
  }
  // now compute corrected image in its floating-point representation
  Mat correct32f = src32f / lightfield32f;
  // now we need to convert the values to the result type,
  // including the scaling operation
  if(dst.depth()==CV_8U)
    correct32f *= 85;
  if(dst.depth()==CV_16U)
    correct32f *= 21843;
  if(dst.depth()==CV_32F)
    correct32f *= 0.33;
  correct32f.convertTo(dst, dst.type());
}
