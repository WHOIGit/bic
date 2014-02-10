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

void interp::distance_map(cv::OutputArray _dst, double altitude, double pitch, double roll, double width, double height, double focal_length) {
  using cv::Mat;
  using cv::Mat_;

  Mat dst = _dst.getMat();
  dst = Mat::zeros(dst.size(), CV_32F);

  // geometry

  // definitions

  // f              // focal length
  float f = focal_length;

  // Of = [0 0 0]   // focal point
  // Oi = [0 0 -f]  // center of sensor
  // Os = [0 0 alt] // intersection of substrate with Of -> Oi

  // basis vectors of substrate plane (showing values prior to rotation)
  //
  // ei = [a b c] = [1 0 0]
  // ej = [l m n] = [0 1 0]

  Mat ei = (Mat_<float>(3,1) << 1, 0, 0);
  Mat ej = (Mat_<float>(3,1) << 0, 1, 0);

  // Li = [i j -f]  // location of pixel on sensor

  // generate matrices of i and j coordinates for elementwise ops
  //
  // this is equivalent to the following Python/numpy code
  // Ai = np.linspace( -width/2, width/2, xres)
  // Aj = np.linspace(-height/2,height/2, yres)
  // I,J = np.meshgrid(Ai,Aj)

  int xres = dst.size().width;
  int yres = dst.size().height;

  Mat I(xres, yres, CV_32F);
  Mat J(xres, yres, CV_32F);

  double istep = width / (xres-1);
  double jstep = height / (yres-1);

  double i = -width/2;
  for(int x = 0; x < xres; x++, i += istep) {
    float j = -height/2;
    for(int y = 0; y < yres; y++, j += jstep) {
      I.at<float>(y,x) = i;
      J.at<float>(y,x) = j;
    }
  }

  // apply pitch / roll rotations

  // compute rotation matrices
  // pitch
  float sinp = sin(pitch);
  float cosp = cos(pitch);

  Mat Rp = (Mat_<float>(3,3) <<
	    1,    0,    0,
	    0, cosp,-sinp,
	    0, sinp, cosp);

  // roll
  float sinr = sin(roll);
  float cosr = cos(roll);

  Mat Rr = (Mat_<float>(3,3) <<
	    cosr, 0,-sinr,
	    0,    1,    0,
	    sinr, 0, cosr);

  // perform rotations in pitch, roll order
  Mat abc = Rr * (Rp * ei);
  Mat lmn = Rr * (Rp * ej);

  // assign to variables
  float a = abc.at<float>(0,0);
  float b = abc.at<float>(1,0);
  float c = abc.at<float>(2,0);

  float l = lmn.at<float>(0,0);
  float m = lmn.at<float>(1,0);
  float n = lmn.at<float>(2,0);

  // location of pixel on substrate Ls is intersection of
  // substrate plane with ray from Li through Of ([0 0 0])
  //
  // Ls = u(Li-Of) = uLi = Os + vei + wej

  // solving for (u v w)
  // 
  // | u |   | -i  a  l |-1 |  0 |
  // | v | = | -j  b  m |   |  0 |
  // | w |   |  f  c  n |   |-alt|
  //
  // expanding
  //
  //                -alt (am - lb)
  // u = ----------------------------------------
  //     -i (bn - mc) + j (an - lc) + f (am - lb)

  Mat U = -altitude * (a*m - l*b) /
    (-I.mul(b*n - m*c) + J.mul(a*n - l*c) + f * (a*m - l*b));

  // distance to substrate d = ||uLi|| 

  // uLi is (UI,UJ,U-f) so D=||Li|| is sqrt(UI^2 + UJ^2 + Uf^2)
  Mat UI2, UJ2, Uf2;
  cv::pow(U.mul(I),2,UI2);
  cv::pow(U.mul(J),2,UJ2);
  cv::pow(U*f,2,Uf2);
  cv::sqrt(UI2 + UJ2 + Uf2, dst);
}

void interp::dist_weight(cv::Mat D, cv::OutputArray _dst, double delta, int i) {
  using cv::Mat;
  // discretize a distance map D in the distance dimension as
  // weights on a set of equally-spaced distance components
  //    
  // wi = 1 - | D/delta - i | if i - 1 < D/delta < i + 1
  //      0                   otherwise
  _dst.create(D.size(), D.type());
  Mat dst = _dst.getMat();
  dst = Mat::zeros(dst.size(), CV_32F);
  double minD, maxD;
  cv::minMaxLoc(D, &minD, &maxD);
  // test that i - 1 < D/delta < i + 1 is true anywhere
  // for this component
  if(minD/delta >= i + 1 || maxD/delta <= i - 1) {
    return;
  }
  std::cout << i << std::endl; // FIXME debug
  // if so compute 1 - | D/delta - i |
  dst = 1 - cv::abs(D/delta - i);
  // set to 0 where < 0
  cv::threshold(dst, dst, 0, 1, cv::THRESH_TOZERO);
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
