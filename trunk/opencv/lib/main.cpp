#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/opencv.hpp>
#include "prototype.hpp"
#include "interpolation.hpp"

using namespace std;

int main(int argc, char **argv) {
  /*
  if(string(argv[1])=="learn") {
    learn_prototype();
  } else {
    correct_prototype();
  }
  */
  LinearBinning<double> bins = LinearBinning<double>(0,2,0.15);
  typedef vector<pair<double, double> > interp_t;
  interp_t result = bins.interpolate(1.234);
  interp_t::iterator it = result.begin();
  for(; it != result.end(); ++it) {
    pair<double,double> p = *it;
    cout << p.first << ": " << p.second << endl;
  }
}
