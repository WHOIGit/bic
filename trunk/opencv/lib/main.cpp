#include <iostream>
#include <string>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "demosaic.hpp"
#include "Lightfield.hpp"

using namespace std;
using namespace cv;

#define PATH_FILE "paths.txt"
#define OUT_DIR "out"

int main(int argc, char** argv) {
  ifstream inpaths(PATH_FILE);
  string path;
  Lightfield model;
  if(string(argv[1]) == "learn") {
    while(getline(inpaths,path)) {
      cout << "Learning " << path << endl;
      Mat cfa_LR = imread(path, CV_LOAD_IMAGE_ANYDEPTH);
      model.addImage(cfa_LR);
    }
    model.save("model.tiff");
  } else if(string(argv[1]) == "correct") {
    model.load("model.tiff");
    int count = 0;
    ifstream inpaths2(PATH_FILE);
    while(getline(inpaths2,path)) {
      cout << "Correcting " << path << endl;
      Mat cfa_LR = model.correct(imread(path, CV_LOAD_IMAGE_ANYDEPTH));
      cout << "Demosaicing " << path << endl;
      Mat rgb_LR = demosaic_hqlinear(cfa_LR,"rggb");
      stringstream outpaths;
      string outpath;
      outpaths << "out/correct" << count << ".tiff";
      outpath = outpaths.str();
      cout << "Saving RGB to " << outpath << endl;
      Mat rgb_LR_16u;
      rgb_LR.convertTo(rgb_LR_16u, CV_16U);
      imwrite(outpath, rgb_LR_16u);
      count++;
    }
  }
  return 0;
}

