#pragma once
#include <exception>
#include <opencv2/opencv.hpp>

#include <iostream>
// FIXME remove iostream

using namespace cv;
using namespace std;

class Lightfield {
  Mat sum;
  Mat count;
  Mat alpha_ones;
  void init(Mat image) {
    if(sum.empty()) {
      int h = image.size().height;
      int w = image.size().width;
      sum = Mat::zeros(h, w, CV_32F);
      count = Mat::zeros(h, w, CV_32F);
      alpha_ones = Mat::ones(h, w, CV_32F);
    }
  }
public:
  Lightfield() { }
  void addImage(Mat image, Mat alpha) {
    // initialize sum/count images
    init(image);
    assert(image.size().height == sum.size().height);
    assert(image.size().width == sum.size().width);
    assert(image.size().height == alpha.size().height);
    assert(image.size().width == alpha.size().width);
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
  Mat correct(Mat image) {
    Mat average = getAverage();
    assert(image.size().height == average.size().height);
    assert(image.size().width == average.size().width);
    double minLightmap, maxLightmap;
    minMaxLoc(average, &minLightmap, &maxLightmap);
    // perform correction
    Mat image32f;
    image.convertTo(image32f, CV_32F);
    return (image32f / average) * (maxLightmap - minLightmap);
  }
  void save(string pathname) {
    // FIXME assert that the path ends ".tif" or equiv
    Mat average = getAverage();
    Mat composite;
    int h = average.size().height;
    int w = average.size().width;
    composite.create(h * 2, w, CV_32F);
    Mat top(composite, Rect(0, 0, w, h));
    Mat bottom(composite, Rect(0, h, w, h));
    average.copyTo(top);
    count.copyTo(bottom);
    Mat composite_16u;
    composite.convertTo(composite_16u, CV_16U);
    imwrite(pathname, composite_16u);
  }
  void load(string pathname) {
    Mat composite_16u = imread(pathname, CV_LOAD_IMAGE_ANYDEPTH);
    assert(composite_16u.type() == CV_16U);
    int h = composite_16u.size().height / 2;
    int w = composite_16u.size().width;
    Mat composite;
    composite_16u.convertTo(composite, CV_32F);
    Mat top(composite, Rect(0, 0, w, h));
    Mat bottom(composite, Rect(0, h, w, h));
    init(top); // initialize accumulation matricies
    top.copyTo(sum);
    bottom.copyTo(count);
    sum = sum.mul(count);
  }
};
