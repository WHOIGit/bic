#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/opencv.hpp>
#include "prototype.hpp"
#include "interpolation.hpp"
#include "demosaic.hpp"

using namespace std;

void demoBinning() { // FIXME delete
  interp::LinearBinning<int> bins = interp::LinearBinning<int>(0,100,7);
  typedef vector<pair<int, double> > interp_t;
  interp_t result = bins.interpolate(31);
  interp_t::iterator it = result.begin();
  for(; it != result.end(); ++it) {
    pair<int,double> p = *it;
    cout << p.first << ": " << p.second << endl;
  }
}

void demoSmoothing(int argc, char **argv) { // FIXME delete
  using namespace cv;
  Mat cfa = imread(argv[1],CV_LOAD_IMAGE_ANYDEPTH);
  Mat q(cfa.size(), cfa.type());
  cfa_quad(cfa,q);
  assert(!q.empty());
  imwrite("cfa_quad.tiff",q);
  quad_cfa(q,cfa);
  cfa_smooth(cfa,cfa,31);
  imwrite("cfa_smooth.tiff",cfa);
  imwrite("rgb_smooth.tiff",demosaic(cfa,"rggb"));
}

int main(int argc, char **argv) {
  if(string(argv[1])=="learn") {
    learn_prototype();
  } else {
    correct_prototype();
  }
}
