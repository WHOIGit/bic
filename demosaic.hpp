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
 * @param src the CFA image
 * @param dst the channels mosaic (same size and type or will be
 * created)
 */
void cfa_quad(cv::InputArray src, cv::OutputArray dst);

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
 * @param src the channels mosaic 
 * @param dst the CFA image (same size and type or will be
 * created)
 */
void quad_cfa(cv::InputArray quad, cv::OutputArray dst);

/**
 * Return a half-resolution image containing pixels at the given
 * Bayer offset.
 *
 * @param src the CFA image
 * @param dst the output image (must be half the resolution)
 * @param x the x offset (0 or 1, default 0)
 * @param y the y offset (0 or 1, default 0)
 *
 * @return the channel image
 */
void cfa_channel(cv::InputArray src, cv::OutputArray dst, int x=0, int y=0);

/**
 * Smooth a Bayer-patterned image with a Gaussian filter, applying
 * the filter to the four Bayer offsets independently.
 *
 * @param cfa the Bayer-patterned image
 *
 * @param ksize the kernel size. Use approximately half the kernel
 * size you would use for a full-resolution image. Must be odd. Sigma
 * will be computed from it using the following formula:
 *
 * 0.3*((ksize-1)*0.5 - 1) + 0.8;
 *
 * @param src the CFA image
 * @param dst the output image
 */
void cfa_smooth(cv::InputArray src, cv::OutputArray dst, int ksize);

