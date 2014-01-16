#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/opencv.hpp>
#include "prototype.hpp"
#include "interpolation.hpp"
#include "illumination.hpp"
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

void smoothAnalysis() {
  using namespace cv;
  for(int i = 0; i <= 8; i++) {
    stringstream inpaths;
    inpaths << "out/slice_" << i << ".tiff";
    //1360x1024
    Mat cfa(imread(inpaths.str(), CV_LOAD_IMAGE_ANYDEPTH), Rect(0,0,1360,1024));
    //Mat roi(cfa, Rect(0,0,100,100));
    for(int j = 0; j < 5; j++) {
      stringstream outpaths;
      Mat new_cfa = cfa.clone();
      cfa_smooth(cfa, new_cfa, 17);
      outpaths << "sluice_" << i << j << ".tiff";
      Mat diff = Mat(cfa.size(), cfa.type());
      illum::correct(cfa, diff, new_cfa);
      string outpath = outpaths.str();
      cout << outpath << endl;
      imwrite(outpath,diff);
      new_cfa.copyTo(cfa);
    }
    cout << endl;
  }
}
int main(int argc, char **argv) {
  if(string(argv[1])=="learn") {
    learn_prototype();
  } else {
    correct_prototype();
  }
}
