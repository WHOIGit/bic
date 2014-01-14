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

/**
 * Generate a mosaic of four half-resolution images containing pixels from
 * each Bayer offset, i.e. an image laid out like this with respect to
 * Bayer offsets x,y:
 *
 * +---+---+
 * |0,0|1,0|
 * +---+---+
 * |0,1|1,1|
 * +---+---+
 *
 * @param cfa the image
 */
cv::Mat cfa_quad(cv::Mat cfa);

/**
 * Given a mosaic of four half-resolution images containing pixels from
 * each Bayer offset, i.e. an image laid out like this with respect to
 * Bayer offsets x,y:
 *
 * +---+---+
 * |0,0|1,0|
 * +---+---+
 * |0,1|1,1|
 * +---+---+
 *
 * produce the full-resolution CFA image. This is the inverse operation
 * of cfa_quad.
 *
 * @param cfa the image mosaic
 */
cv::Mat quad_cfa(cv::Mat quad);

/**
 * Return a half-resolution image containing pixels at the given
 * Bayer offset.
 *
 * @param cfa the image
 * @param x the x offset (0 or 1, default 0)
 * @param y the y offset (0 or 1, default 0)
 */
cv::Mat cfa_channel(cv::Mat cfa, int x=0, int y=0);
