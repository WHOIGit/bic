#pragma once
#include <opencv2/opencv.hpp>

/**
 * Demosaic a color-filter-array (a.k.a. "RAW") image and produce
 * a three-channel color image (BGR).
 *
 * This function implements Malvar et al's "high quality linear"
 * algorithm.
 *
 * http://research.microsoft.com/apps/pubs/default.aspx?id=102068
 *
 * Any input type is supported. The output type will match the input
 * type. Internally, all computations are done in 32-bit floating
 * point.
 *
 * @param cfa the color filter array (CFA) patterned image
 *
 * @param cfaPattern a string describing the Bayer pattern; one of
 * "rggb", "bggr", "grbg", or "gbrg". Case insensitive.
 *
 * @return a new color (BGR) image
 */
cv::Mat demosaic(cv::Mat cfa, std::string cfaPattern="rggb");
