#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/opencv.hpp>
#include <boost/tokenizer.hpp>
#include "disparity.hpp"
#include "demosaic.hpp"

#define PATH_FILE "alts.txt"

using namespace std;
using namespace cv;

int main(int argc, char **argv) {
  ifstream inpaths(PATH_FILE);
  string line;
  while(getline(inpaths,line)) {
    boost::char_separator<char> sep(",");
    boost::tokenizer< boost::char_separator<char> >tokens(line,sep);
    string inpath = *(tokens.begin());
    //cout << "loading " << inpath << endl;
    Mat cfa_LR = imread(inpath, CV_LOAD_IMAGE_ANYDEPTH);
    //cout << "loaded " << inpath << endl;
    Mat bgr_LR = demosaic(cfa_LR,"rggb");
    //cout << "demosaiced " << inpath << endl;
    // now convert to fixed point
    bgr_LR /= 256;
    bgr_LR.convertTo(bgr_LR, CV_8U);
    Mat y_LR;
    cvtColor(bgr_LR, y_LR, CV_BGR2GRAY);
    // first compute disparity and output disparity image
    /*
    Mat disp = Mat::zeros(y_LR.size().height, y_LR.size().width/2, CV_16S);
    disparity(get_L(y_LR), get_R(y_LR), disp);
    imwrite("disparity.tiff",disp);
    */
    // now we can do stereo correspondence
    int x = xOffset(y_LR);
    cout << inpath << "," << x << endl;
    // now generate the overlay
    /*
    int w = y_LR.size().width / 2;
    int h = y_LR.size().height;
    Mat overlay = Mat::zeros(h,w*2,y_LR.type());
    Rect left_roi = Rect(0,0,w,h);
    Rect right_roi = Rect(w,0,w,h);
    Rect offset_roi = Rect(0-x,0,w,h);
    Mat left_image(y_LR, left_roi);
    Mat right_image(y_LR, right_roi);
    Mat left(overlay, left_roi);
    Mat right(overlay, offset_roi);
    left += left_image / 2;
    right += right_image / 2;
    imwrite("overlay.tiff",overlay);
    */
  }
}
