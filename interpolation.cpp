#include "interpolation.hpp"

// FIXME assumes that axes of rotation intersect center of image
// FIXME allow for specification of center at aribtrary location
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

  Mat ei = (Mat_<float>(3,1) <<  1, 0, 0);
  Mat ej = (Mat_<float>(3,1) <<  0, 1, 0);

  // Li = [i j -f]  // location of pixel on sensor

  // generate matrices of i and j coordinates for elementwise ops
  //
  // this is equivalent to the following Python/numpy code
  // Ai = np.linspace( -width/2, width/2, xres)
  // Aj = np.linspace(-height/2,height/2, yres)
  // I,J = np.meshgrid(Ai,Aj)

  int xres = dst.size().width;
  int yres = dst.size().height;

  Mat I(yres, xres, CV_32F);
  Mat J(yres, xres, CV_32F);

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

  // apply pitch / roll rotations if necessary

  // compute rotation matrices

  Mat abc = ei;
  Mat lmn = ej;

  // apply pitch before roll, assuming that pitch and roll are small
  // enough to make order-dependency not matter much

  // pitch
  if(pitch != 0) {
    // assuming the top of the frame depicts substrate that is closer
    // when pitch is positive (pitched forward), invert pitch relative
    // to basis vector
    float sinp = sin(0-pitch);
    float cosp = cos(0-pitch);

    Mat Rp = (Mat_<float>(3,3) <<
	      1,    0,    0,
	      0, cosp,-sinp,
	      0, sinp, cosp);

    abc = Rp * abc;
    lmn = Rp * lmn;
  }

  // roll
  if(roll != 0) {
    // assuming that the right of the frame depicts substrate that
    // is closer when roll is positive (pitched to the right), leave
    // roll as is relative to basis vector
    float sinr = sin(roll);
    float cosr = cos(roll);

    Mat Rr = (Mat_<float>(3,3) <<
	      cosr, 0,-sinr,
	      0,    1,    0,
	      sinr, 0, cosr);

    abc = Rr * abc;
    lmn = Rr * lmn;
  }

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
  // if so compute 1 - | D/delta - i |
  dst = 1 - cv::abs(D/delta - i); // FIXME use absdiff
  // set to 0 where < 0
  cv::threshold(dst, dst, 0, 1, cv::THRESH_TOZERO);
}
