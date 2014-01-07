#pragma once
#include <exception>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

class Lightfield {
  Mat sum;
  Mat count;
  Mat alpha_ones;
  void init(Mat image) {
    if(sum.empty()) {
      sum = Mat::zeros(image.rows, image.cols, CV_32F);
      count = Mat::zeros(image.rows, image.cols, CV_32F);
      alpha_ones = Mat::ones(image.rows, image.cols, CV_32F);
    }
  }
public:
  Lightfield() { }
  void addImage(Mat image, Mat alpha) {
    // initialize sum/count images
    init(image);
    assert(image.rows == sum.rows);
    assert(image.cols == sum.cols);
    assert(image.rows == alpha.rows);
    assert(image.cols == alpha.cols);
    Mat image32f;
    image.convertTo(image32f, CV_32F);
    sum += image32f.mul(alpha);
    count += alpha_ones;
  }
  void addImage(Mat image) {
    // assumes alpha image is all ones
    init(image);
    addImage(image, alpha_ones);
  }
  Mat getAverage() {
    assert(!sum.empty());
    return sum / count;
  }
  // FIXME add save/load
  Mat correct(Mat image) {
    Mat average = getAverage();
    assert(image.rows == average.rows);
    assert(image.cols == average.cols);
    double minLightmap, maxLightmap;
    minMaxLoc(average, &minLightmap, &maxLightmap);
    // perform correction
    Mat image32f;
    image.convertTo(image32f, CV_32F);
    return (image32f / average) * (maxLightmap - minLightmap);
  }
};
