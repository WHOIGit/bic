#include "illumination.hpp"

/**
 * Correct an image by dividing it by an illumination model image,
 * e.g., one computed by frame averaging
 *
 * @param src the source image
 * @param dst the destination image (must be same size and type as src)
 * @return a new, correctly-illuminated image (in floating point)
 */
void illum::correct(cv::InputArray _src, cv::OutputArray _dst, cv::Mat lightfield) {
  using namespace cv;
  Mat src = _src.getMat();
  _dst.create(src.size(), src.type());
  Mat dst = _dst.getMat();
  if(src.size() != dst.size())
    throw std::runtime_error("input/output image size mismatch");
  if(src.type() != dst.type())
    throw std::runtime_error("input/output image type mismatch");
  if(src.size() != lightfield.size())
    throw std::runtime_error("input image not same size as lightfield");
  Mat src32f;
  src.convertTo(src32f, CV_32F); // convert to floating point
  // find intensity range of lightfield image
  double minLightmap, maxLightmap;
  minMaxLoc(lightfield, &minLightmap, &maxLightmap);
  // divide image by lightfield, then normalize to intensity range of lightfield
  Mat lightfield32f;
  if(lightfield.type() != CV_32F) {
    lightfield.convertTo(lightfield32f, CV_32F);
  } else {
    lightfield32f = lightfield;
  }
  Mat correct32f = (src32f / lightfield32f) * (maxLightmap - minLightmap);
  correct32f.convertTo(dst, src.type());
}
