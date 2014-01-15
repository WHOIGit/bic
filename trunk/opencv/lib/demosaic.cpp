#include <opencv2/opencv.hpp>
#include <boost/algorithm/string.hpp>
#include "demosaic.hpp"

using namespace cv;
using namespace std;

string str_tolower(string in) { // FIXME dead code
  string out;
  out.append(in);
  transform(out.begin(), out.end(), out.begin(), ::tolower);
  return out;
}

Mat cfa_kern(string cfa, char chan) {
  int A = cfa[0]==chan;
  int B = cfa[1]==chan;
  int C = cfa[2]==chan;
  int D = cfa[3]==chan;
  return (Mat_<uchar>(2,2) <<
	  A, B,
	  C, D);
}

Mat r_kern(string cfa) {
  return cfa_kern(cfa,'r');
}
Mat g_kern(string cfa) {
  return cfa_kern(cfa,'g');
}
Mat b_kern(string cfa) {
  return cfa_kern(cfa,'b');
}
// R at G, R row as well as B at G, B col
Mat ratg_rrow_kern(string cfa) {
  int A = cfa[0]=='g' && cfa[1]=='r';
  int B = cfa[1]=='g' && cfa[0]=='r';
  int C = cfa[2]=='g' && cfa[3]=='r';
  int D = cfa[3]=='g' && cfa[2]=='r';
  return (Mat_<uchar>(2,2) <<
	  A, B,
	  C, D);
}
// R at G, R col as well as B at G, B row
Mat ratg_rcol_kern(string cfa) {
  int A = cfa[0]=='g' && cfa[2]=='r';
  int B = cfa[1]=='g' && cfa[3]=='r';
  int C = cfa[2]=='g' && cfa[0]=='r';
  int D = cfa[3]=='g' && cfa[1]=='r';
  return (Mat_<uchar>(2,2) <<
	  A, B,
	  C, D);
}

Mat demosaic(Mat image_in, string cfaPattern) {
  // "High Quality Linear" (Malvar et al)
  // convert to floating point, if necessary
  Mat image;
  if(image.depth()==CV_32F) {
    image = image_in;
  } else {
    image_in.convertTo(image, CV_32F);
  }
  // perform no conversion of values

  // collect metrics
  int h, w;
  Size S = image.size();
  h = S.height;
  w = S.width;

  // Bayer pattern
  boost::to_lower(cfaPattern);

  // construct G channel
  Mat G;
  G.create(S, CV_32F);

  // first, composite existing G pixels into G channel

  // construct G mask
  Mat gmask = repeat(g_kern(cfaPattern), h/2, w/2);
  // copy G data into output image
  image.copyTo(G, gmask);

  // now interpolate rest of G pixels
  Mat cfa2G = (Mat_<float>(5,5) <<
	        0, 0,-1, 0, 0,
 	        0, 0, 2, 0, 0,
	       -1, 2, 4, 2,-1,
 	        0, 0, 2, 0, 0,
 	        0, 0,-1, 0, 0) / 8;
  Mat iG;
  filter2D(image, iG, CV_32F, cfa2G);
  iG.copyTo(G, 1-gmask);

  // now, R/B at B/R locations

  // construct channels
  Mat R, B;
  R.create(S, CV_32F);
  B.create(S, CV_32F);

  // construct masks
  Mat bmask = repeat(b_kern(cfaPattern), h/2, w/2);
  Mat rmask = repeat(r_kern(cfaPattern), h/2, w/2);

  // RB at RB locations from original image data
  image.copyTo(R, rmask);
  image.copyTo(B, bmask);

  // interpolate RB at BR locations
  Mat rb2br = (Mat_<float>(5,5) <<
	      0, 0, -1.5, 0,    0,
              0, 2,    0, 2,    0,
	   -1.5, 0,    6, 0, -1.5,
	      0, 2,    0, 2,    0,
   	      0, 0, -1.5, 0,    0) / 8;
  // R at B locations
  Mat iRB;
  filter2D(image, iRB, CV_32F, rb2br);
  iRB.copyTo(B, rmask);
  iRB.copyTo(R, bmask);

  // RB at G in RB row, BR column
  Mat rbatg_rbrow = (Mat_<float>(5,5) <<
		    0,  0, 0.5,  0,  0,
		    0, -1,   0, -1,  0,
		   -1,  4,   5,  4, -1,
		    0, -1,   0, -1,  0,
 		    0,  0, 0.5,  0,  0) / 8;
  // RB at G in BR row, RB column
  Mat rbatg_rbcol = (Mat_<float>(5,5) <<
		      0,  0, -1,  0,   0,
		      0, -1,  4, -1,   0,
		    0.5,  0,  5,  0, 0.5,
 		      0, -1,  4, -1,   0,
 		      0,  0, -1,  0,   0) / 8;

  // construct masks
  Mat ratg_rrow_mask = repeat(ratg_rrow_kern(cfaPattern), h/2, w/2);
  Mat ratg_rcol_mask = repeat(ratg_rcol_kern(cfaPattern), h/2, w/2);
  Mat batg_brow_mask = ratg_rcol_mask;
  Mat batg_bcol_mask = ratg_rrow_mask;

  filter2D(image, iRB, CV_32F, rbatg_rbrow);
  // RB at G in RB row, BR column
  iRB.copyTo(R,ratg_rrow_mask);
  iRB.copyTo(B,batg_brow_mask);

  // RB at G in BR row, RB column
  filter2D(image, iRB, CV_32F, rbatg_rbcol);
  iRB.copyTo(R,ratg_rcol_mask);
  iRB.copyTo(B,batg_bcol_mask);

  // now convert output back to original image depth
  Mat color, out;
  vector<Mat> BGR(3);
  BGR[0] = B;
  BGR[1] = G;
  BGR[2] = R;
  merge(BGR, color);
  color.convertTo(out, image_in.depth());
  return out;
}

/// utility

void cfa_quad(Mat src, Mat dst) {
  assert(dst.size()==src.size());
  assert(dst.type()==src.type());
  Mat xMap, yMap;
  xMap.create(src.size(), CV_32F);
  yMap.create(src.size(), CV_32F);
  // build map
  int c2 = src.cols/2;
  int r2 = src.rows/2;
  for(int x = 0; x < src.cols; x++) {
    for(int y = 0; y < src.rows; y++) {
      if(x < c2) {
	xMap.at<float>(y,x) = x * 2;
      } else {
	xMap.at<float>(y,x) = ((x - c2) * 2) + 1;
      }
      if(y < r2) {
	yMap.at<float>(y,x) = y * 2;
      } else {
	yMap.at<float>(y,x) = ((y - r2) * 2) + 1;
      }
    }
  }
  Mat remapped(src.size(), src.type());
  remap(src,remapped,xMap,yMap,INTER_NEAREST);
  remapped.copyTo(dst);
}

void quad_cfa(Mat src, Mat dst) {
  assert(dst.size()==src.size());
  assert(dst.type()==src.type());
  Mat xMap, yMap;
  xMap.create(src.size(), CV_32F);
  yMap.create(src.size(), CV_32F);
  // build map
  int c2 = src.cols/2;
  int r2 = src.rows/2;
  for(int x = 0; x < src.cols; x++) {
    for(int y = 0; y < src.rows; y++) {
      if(x % 2 == 0) {
	xMap.at<float>(y,x) = x / 2;
      } else {
	xMap.at<float>(y,x) = c2 + ((x-1) / 2);
      }
      if(y % 2 == 0) {
	yMap.at<float>(y,x) = y / 2;
      } else {
	yMap.at<float>(y,x) = r2 + ((y-1) / 2);
      }
    }
  }
  Mat remapped(src.size(), src.type());
  remap(src,remapped,xMap,yMap,INTER_NEAREST);
  remapped.copyTo(dst);
}

void cfa_channel(Mat src, Mat dst, int x, int y) {
  assert(x==0 || x==1);
  assert(y==0 || y==1);
  assert(dst.type()==src.type());
  int w2 = src.cols/2;
  int h2 = src.rows/2;
  assert(dst.cols==w2);
  assert(dst.rows==h2);
  // now return just the requested quadrant
  Mat quad(src.size(), src.type());
  cfa_quad(src,quad);
  Mat roi(quad, Rect(w2*x,h2*y,w2,h2));
  roi.copyTo(dst);
}

void cfa_smooth(Mat src, Mat dst, int ksize) {
  assert(dst.size()==src.size());
  assert(dst.type()==src.type());
  int w2 = src.size().width / 2;
  int h2 = src.size().height / 2;
  cfa_quad(src,dst);
  double sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8;
  for(int x = 0; x < 2; x++) {
    for(int y = 0; y < 2; y++) {
      Rect roi = Rect(x*w2,y*h2,w2,h2);
      Mat q(dst, roi);
      GaussianBlur(q,q,Size(ksize,ksize),sigma);
    }
  }
  quad_cfa(dst,dst);
}
