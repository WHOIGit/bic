#pragma once

namespace learn_correct {
  void learn();
  void correct();
}

#define PATH_FILE "aprs.csv" // FIXME hardcoded CSV file path
#define BAYER_PATTERN "rggb" // FIXME hardcoded bayer pattern
#define OUT_DIR "out" // FIXME hardcoded model/output directory
#define N_THREADS 12 // FIXME hardcoded thread count
#define ALT_SPACING_M 0.1 // FIXME hardcoded altitude bin spacing
#define FOCAL_LENGTH_M 0.012 // FIXME hardcoded focal length
#define PIXEL_SEP_M 0.0000065 // FIXME hardcoded pixel separation
