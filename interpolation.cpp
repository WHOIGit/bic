#include "interpolation.hpp"

// FIXME does this handle 0 pitch/roll correctly?
// FIXME does this correctly convert back to meters at the end?
cv::Mat interp::alt_pitch_roll(float altitude, float pitch, float roll, int width, int height, int xres, int yres, float focal_length, float pixel_sep) {
  using cv::Mat;
  // to enable element-wise operations, construct matricies of x and y
  // distance from center of frame, in pixels, at specified xres/yres
  Mat X(yres, xres, CV_32F);
  Mat Y(yres, xres, CV_32F);
  float x = 0 - width/2; // x distance from center of frame in pixels
  float xstep = (float)width / xres; // and its increment at this resolution
  for(int i = 0; i < xres; i++, x += xstep) {
    float y = 0 - height/2; // y distance from center of frame in pixels
    float ystep = (float)height / yres; // and its increment at this resolution
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
  double tan_roll = tan(0-roll); // clockwise around x axis
  Mat xd = (ta + tan_roll) / (1.0 - (ta * tan_roll));
  // y distance map based on pitch
  Mat tb = Y / efl; // y distance from center in units of focal length
  double tan_pitch = tan(pitch); // clockwise around y axis
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

// here is a port of JRock's alt/pitch/roll code that uses a cv::Mat for output instead of a vil_image_view,
// and accepts the following params
// - altitude in meters
// - pitch in degrees
// - roll in degrees
// - width of frame in pixels
// - height of frame in pixels
// - focal length *in pixels*
cv::Mat jrock_calculate_altitude_map(float alt, float pitch, float roll, int width, int height, float focal_length) {
  // this member function calculates the height of every pixel in the image and stores it in "output"
  // the HabCam Teledyne PSA-916 altimeter uses a wide-beam first-return signal, so altitude is perpendicular to substrate

  // JMF: create output matrix at image size and at double-precision floating point
  cv::Mat output(height, width, CV_64F);
  // JMF: JRock appears to have transposed "x" and "y" and so I will use that notation here

  // get the image size to calculate coordinate offset to center
  //int xbar = output.ni()/2;
  //int ybar = output.nj()/2;
  // JMF using OpenCV rows and cols here
  int xbar = output.rows/2; // JMF: JRock calls the along-the-height axis "x"
  int ybar = output.cols/2; // JMF: JRock calls the along-the-width axis "y"

  //Note that roll is opposite relative to image coords
  double tan_pitch = tan(3.14*pitch/180.0);
  double tan_roll = tan(-3.14*roll/180.0);

  // loop through all the pixels positions for the defined image size
  for(unsigned x = 0; x<output.rows; ++x){ // JMF using OpenCV rows
    for(unsigned y = 0; y<output.cols; ++y){ // JMF using OpenCV cols

      // translate coordinate by distance-to-center such that (0,0) is at the image center (at focal axis)
      int xhat = int(x)-xbar;
      int yhat = int(y)-ybar;

      double tanalpha = double(xhat)/focal_length;
      double xdist = (tanalpha+tan_pitch)/(1-tanalpha*tan_pitch);

      double tanbeta = double(yhat)/focal_length;
      double ydist = (tanbeta+tan_roll)/(1-tanbeta*tan_roll);

      double distance = alt * sqrt(1 + xdist*xdist + ydist*ydist);

      output.at<double>(x,y) = distance*focal_length/sqrt(focal_length*focal_length + xhat*xhat + yhat*yhat);
    }
  }

  return output;
}
