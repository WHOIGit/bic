#include "illumination.hpp"

/**
 * Correct an image by dividing it by an illumination model image,
 * e.g., one computed by frame averaging
 *
 * @param image the image
 * @return a new, correctly-illuminated image
 */
cv::Mat illum::correct(cv::Mat image, cv::Mat lightfield) {
  assert(image.size().height == lightfield.size().height);
  assert(image.size().width == lightfield.size().width);
  cv::Mat image32f;
  image.convertTo(image32f, CV_32F); // convert to floating point
  // find intensity range of lightfield image
  double minLightmap, maxLightmap;
  cv::minMaxLoc(lightfield, &minLightmap, &maxLightmap);
  // divide image by lightfield, then normalize to intensity range of lightfield
  return (image32f / lightfield) * (maxLightmap - minLightmap);
}
