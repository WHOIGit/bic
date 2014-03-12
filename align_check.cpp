#include <stdio.h>
#include <stdexcept>
#include <opencv2/opencv.hpp>

#include "stereo.hpp"
#include "demosaic.hpp"

#define CAMERA_SEP 0.235
#define FOCAL_LENGTH 0.012
#define H2O_ADJUSTMENT 1.25
#define PIXEL_SEP 0.00000645
#define TEMPLATE_SIZE 64

/* open and display an image */
int main( int argc, char** argv )
{
  using namespace cv;
  using stereo::align;
  using namespace std;
  Mat cfa_LR;
  cfa_LR = imread( argv[1], CV_LOAD_IMAGE_ANYDEPTH );

  // now, compute offset

  // compute from parallax
  // pull green channel assuming rggb
  Mat Gi;
  cfa_channel(cfa_LR, Gi, 1, 0);
  Gi *= 255.0 / 65535.0;
  Mat G;
  Gi.convertTo(G, CV_8U);

  // compute pixel offset
  int xoff = align(G, TEMPLATE_SIZE);
  if(xoff <= 0) // bad alignment
    throw std::runtime_error("unable to compute altitude from parallax");
  // convert to meters
  double alt = (CAMERA_SEP * FOCAL_LENGTH * H2O_ADJUSTMENT) / ((xoff*2) * PIXEL_SEP);

  cout << "offset = " << xoff << endl;
  cout << "alt = " << alt << endl;

  Mat xMap, yMap;
  xMap.create(G.size(), CV_32F);
  yMap.create(G.size(), CV_32F);
  int c2 = G.cols/2;
  for(int x = 0; x < G.cols; x++) {
    for(int y = 0; y < G.rows; y++) {
      yMap.at<float>(y,x) = y;
      if(x < c2) {
	xMap.at<float>(y,x) = x + c2 - xoff;
      } else {
	xMap.at<float>(y,x) = -1;
      }
    }
  }
  Mat Go;
  remap(G,Go,xMap,yMap,INTER_NEAREST);

  namedWindow( "Display Image", CV_WINDOW_AUTOSIZE );
  imshow( "Display Image", G + Go );

  waitKey(0);

  return 0;
}
