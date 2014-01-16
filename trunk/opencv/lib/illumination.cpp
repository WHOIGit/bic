#include "illumination.hpp"

/**
 * Correct an image by dividing it by an illumination model image,
 * e.g., one computed by frame averaging
 *
 * @param src the source image
 * @param dst the destination image (must be same size and type as src)
 * @return a new, correctly-illuminated image (in floating point)
 */
void illum::correct(cv::Mat src, cv::Mat dst, cv::Mat lightfield) {
  assert(src.size()==dst.size());
  assert(src.type()==dst.type());
  assert(src.size()==lightfield.size());
  cv::Mat src32f;
  src.convertTo(src32f, CV_32F); // convert to floating point
  // find intensity range of lightfield image
  double minLightmap, maxLightmap;
  cv::minMaxLoc(lightfield, &minLightmap, &maxLightmap);
  // divide image by lightfield, then normalize to intensity range of lightfield
  cv::Mat lightfield32f;
  if(lightfield.type() != CV_32F) {
    lightfield.convertTo(lightfield32f, CV_32F);
  } else {
    lightfield32f = lightfield;
  }
  cv::Mat correct32f = (src32f / lightfield32f) * (maxLightmap - minLightmap);
  correct32f.convertTo(dst, src.type());
}
