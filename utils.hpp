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
}
