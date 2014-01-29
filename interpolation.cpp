#include "interpolation.hpp"

// FIXME does this handle 0 pitch/roll correctly?
// FIXME does this correctly convert back to meters at the end?
cv::Mat alt_pitch_roll(float altitude, float pitch, float roll, int width, int height, int xres, int yres, float focal_length, float pixel_sep) {
  using cv::Mat;
  // to enable element-wise operations, construct matricies of x and y
  // distance from center of frame, in pixels, at specified xres/yres
  Mat X(yres, xres, CV_32F);
  Mat Y(yres, xres, CV_32F);
  float x = 0 - width/2; // x distance from center of frame in pixels
  float xstep = (float)width / xres; // and its increment at this resolution
  for(int i = 0; i < xres; i++, x += xstep) {
    float y = 0 - height/2; // y distance from center of frame in pixels
    float ystep = (float)width / yres; // and its increment at this resolution
    for(int j = 0; j < yres; j++, y += ystep) {
      X.at<float>(j,i) = x;
      Y.at<float>(j,i) = y;
    }
  }
  // pixel_sep is distance between centers of pixels at sensor, in meters
  // focal length is in meters
  // convert focal length to pixels
  float efl = focal_length / pixel_sep;
  // x distance map based on roll
  Mat ta = X / efl; // x distance from center in units of focal length
  double tan_roll = tan(roll); // clockwise around x axis
  Mat xd = (ta + tan_roll) / (1.0 - (ta * tan_roll));
  // y distance map based on pitch
  Mat tb = Y / efl; // y distance from center in units of focal length
  double tan_pitch = tan(0-pitch); // clockwise around y axis
  Mat yd = (tb + tan_pitch) / (1.0 - (tb * tan_pitch));
  // xy distance map
  Mat xd2, yd2;
  cv::pow(xd,2,xd2);
  cv::pow(yd,2,yd2);
  Mat D;
  cv::sqrt(1.0 + xd2 + yd2, D);
  Mat X2, Y2;
  cv::pow(X,2,X2);
  cv::pow(Y,2,Y2);
  Mat eXY;
  cv::sqrt(pow(efl,2) + X2 + Y2, eXY);
  // to convert from pixels back to meters,
  // simplify
  // given efl = focal_length / pixel_sep
  // return (D * (altitude / pixel_sep) * efl / eXY) * pixel_sep
  // as
  // return D * altitude * efl / eXY
  return D * altitude * efl / eXY;
}
