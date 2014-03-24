#pragma once
#include <cmath>
#include <opencv2/opencv.hpp>

/**
 * Utilities to support interpolation between lightmaps for a
 * downward-facing benthic vehicle, based on camera metrics and
 * vehicle orientation.
 *
 * The assumptions that this module makes include:
 *
 * - the camera image plane is parallel to the substate when the vehicle
 * is at zero pitch and roll (e.g., yaw does not matter)
 * - the origin of pitch and roll is located somewhere between the two cameras
 * - pitch and roll are small enough that it does not matter in what order
 * the rotations are applied
 * 
 * See interp::distance_map for more details.
 *
 * @author Joe Futrelle
 * 
 * @brief distance mapping and discretization for lightmap interpolation.
 */
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
   *
   * @author Joe Futrelle
   */
  void dist_weight(cv::Mat D, cv::OutputArray _dst, double delta=1.0, int i=0);
}
