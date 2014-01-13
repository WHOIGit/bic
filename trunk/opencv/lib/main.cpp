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
  LinearBinning<int> bins = LinearBinning<int>(0,100,7);
  typedef vector<pair<int, double> > interp_t;
  interp_t result = bins.interpolate(31);
  interp_t::iterator it = result.begin();
  for(; it != result.end(); ++it) {
    pair<int,double> p = *it;
    cout << p.first << ": " << p.second << endl;
  }
}
