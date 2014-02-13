#pragma once
#include <cmath>
#include <opencv2/opencv.hpp>

namespace interp {
  /**
   * Compute distance to substrate per-pixel given altitude, pitch,
   * roll, and camera metrics (focal length, size of pixel on sensor).
   *
   * The orientation of the image is assumed to be such that pitch is
   * a clockwise rotation around the y axis and roll is a clockwise
   * rotation around the x axis. As a result positive pitch brings the
   * substrate closer to the camera at the top (leading) edge of the
   * frame, and positive roll brings the substrate closer to the
   * camera at the right edge of the frame. If the camera is oriented
   * differently, transpose and/or change the signs of the pitch and
   * roll inputs.
   *
   * The output matrix can be computed at reduced or enhanced
   * resolution and/or a different aspect ratio by passing in an
   * output array with the desired dimensions.
   *
   * @param _dst output matrix at desired resolution
   * @param altitude altitude
   * @param pitch pitch in radians
   * @param roll roll in radians
   * @param width width of sensor
   * @param height height of sensor
   * @param focal_length effective focal length
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

  /**
   * Compute distance to substrate per-pixel given altitude, pitch,
   * roll, and camera metrics (focal length, size of pixel on sensor).
   *
   * The orientation of the image is assumed to be such that pitch is
   * a clockwise rotation around the y axis and roll is a clockwise
   * rotation around the x axis. As a result positive pitch brings the
   * substrate closer to the camera at the top (leading) edge of the
   * frame, and positive roll brings the substrate closer to the
   * camera at the right edge of the frame. If the camera is oriented
   * differently, transpose and/or change the signs of the pitch and
   * roll inputs.
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
   * @param focal_length effective focal length in meters
   * @param pixel_sep distance between pixel centers in meters at the sensor
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
  cv::Mat jrock_pitch_roll(float altitude, float pitch, float roll, int width=1360, int height=1024, int xres=1360, int yres=1024, float focal_length=0.012, float pixel_sep=0.0000065);
}
