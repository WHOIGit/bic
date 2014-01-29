#pragma once
#include <cmath>
#include <opencv2/opencv.hpp>

namespace interp {
  template <typename T> class LinearBinning;
}

template <typename T> class interp::LinearBinning {
private:
  std::vector<T> bins;
public:
  LinearBinning() { }
  LinearBinning(T low, T high, T width) {
    assert(high > low);
    assert(width < (high-low));
    for(T v = low; v <= high; v += width) {
      bins.push_back(v);
    }
  }
  std::vector<T> getBins() {
    return bins;
  }
  std::vector<std::pair<T,double> > interpolate(T value) {
    std::vector<std::pair<T,double> > result;
    typename std::vector<T>::iterator it = bins.begin();
    double binWidth = *(it+1) - *it;
    for(; it != bins.end(); ++it) {
      T diff = std::abs(*it - value);
      if(diff < binWidth) {
	result.push_back(std::pair<T,double>(*it, 1.0 - (diff / binWidth)));
      }
    }
    return result;
  }
};

/**
 * Compute distance to substrate per-pixel given altitude, pitch,
 * roll, and camera metrics (focal length, size of pixel on sensor).
 *
 * The orientation of the image is assumed to be such that pitch
 * is a clockwise rotation around the y axis and roll is a clockwise
 * rotation around the x axis.
 *
 * The output matrix can be computed at reduced resolution and/or a
 * different aspect ratio by setting the xres and yres parameters
 * independently of the height and width parameters.
 *
 * @param altitude altitude in meters
 * @param pitch pitch in radians
 * @param roll roll in radians
 * @param width width of frame in pixels
 * @param height height of frame in pixels
 * @param xres resolution of output matrix along width dimension
 * @param yres resolution of output matrix along height dimension
 * @param focal_length EFL in meters
 * @param pixel_sep distance between real pixel centers in meters
 *
 * @return matrix, at specified resolution, containing distance to
 * substrate in meters
 *
 * The output matrix can be upscaled to full camera resolution to
 * efficiently approximate the per-pixel substrate distance.
 *
 * @author Jason Rock (original)
 * @author Joe Futrelle (OpenCV port)
 */
cv::Mat alt_pitch_roll(float altitude, float pitch, float roll, int width=1360, int height=1024, int xres=1360, int yres=1024, float focal_length=0.012, float pixel_sep=0.0000065);
