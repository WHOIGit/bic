#pragma once

#include "learn_correct.hpp"

namespace utils {
  /**
   * Compute altitude from stereo for a batch of images. The input
   * data is expected to be a list of full pathnames to RAW stereo
   * pairs.
   *
   * The output is CSV records with a pathname followed by an
   * altitude.
   */
  void alt_from_stereo(learn_correct::Params);
  /**
   * View a RAW image. Demosaics the image and adjusts its histogram
   * so it doesn't look black, then displays it on the screen.
   */
  void view_raw(learn_correct::Params);
  /**
   * View an image as a crosseye stereo image
   */
  void view_xeye(learn_correct::Params);
  /**
   * Generate thumbnails of a lightmap
   */
  void thumb_lightmap(learn_correct::Params);
  /**
   * Generate side-by-side letterbodxed image
   */
  void side_by_side(learn_correct::Params);
}
