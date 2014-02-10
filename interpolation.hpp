#pragma once
#include <cmath>
#include <opencv2/opencv.hpp>

namespace interp {
  template <typename T> class LinearBinning;
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

  /**
   * Compute distance to substrate per-pixel given altitude, pitch,
   * roll, and camera metrics (focal length, size of pixel on sensor).
   *
   * // FIXME determine if the following sentence is true:
   * The orientation of the image is assumed to be such that pitch
   * is a clockwise rotation around the y axis and roll is a clockwise
   * rotation around the x axis.
   *
   * The output matrix can be computed at reduced resolution and/or a
   * different aspect ratio by setting the xres and yres parameters
   * independently of the height and width parameters.
   *
   * @param _dst output matrix at desired resolution
   * @param altitude altitude
   * @param pitch pitch in radians
   * @param roll roll in radians
   * @param width width of sensor
   * @param height height of sensor
   * @param focal_length effective focal length
   *
   * The output matrix can be upscaled to full camera resolution to
   * approximate the per-pixel substrate distance.
   *
   * @author Joe Futrelle
   */
  void distance_map(cv::OutputArray _dst, double altitude, double pitch, double roll, double width=0.00884, double height=0.006656, double focal_length=0.012);

  /**
   * Discretize a distance map D in the distance dimension as
   * weights on a set of equally-spaced distance components
   *
   * @param D the distance map
   * @param _dst output array (same size/type as D)
   * @param delta spacing of distance components
   * @param i compute D's weight for component with d = i * delta
   */
  void dist_weight(cv::Mat D, cv::OutputArray _dst, double delta=1.0, int i=0);
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

// FIXME testing against old code
cv::Mat jrock_calculate_altitude_map(float altitude, float pitch, float roll, int width, int height, float focal_length);
