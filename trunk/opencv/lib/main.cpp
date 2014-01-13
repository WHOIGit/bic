#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/opencv.hpp>
#include "prototype.hpp"
#include "interpolation.hpp"

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

int main(int argc, char **argv) {
  if(string(argv[1])=="learn") {
    learn_prototype();
  } else {
    correct_prototype();
  }
}
