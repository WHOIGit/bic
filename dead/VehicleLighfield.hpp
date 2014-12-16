class illum::VehicleLightfield : public illum::MultiLightfield {
private:
  stereo::CameraPair cameras;
public:
  /**
   * Create a multi-altitude lightfield.
   * @param step_m the width of each altitude bin in m
   * @param cameras camera metrics
   */
  VehicleLightfield(double step_m, stereo::CameraPair cameras) {
    alt_step = step_m;
    this->cameras = cameras;
  }
  /**
   * Add an image to the lightfield
   * @param image the image to add
   * @param alt the altitude the image was taken at
   * @param pitch the pitch of the vehicle
   * @param roll the roll of the vehicle
   */
  int addImage(Mat image, double alt, double pitch, double roll) {
    // compute distance map
    int width = image.size().width;
    int height = image.size().height;
    Mat D = Mat::zeros(height, width, CV_32F);
    double width_m = width * cameras.pixel_sep;
    double height_m = height * cameras.pixel_sep;
    if(false) { // FIXME unfinished
      Mat left = Mat(D,cv::Rect(0,0,width/2,height));
      Mat right = Mat(D,cv::Rect(width/2,0,width/2,height));
      // now compute center of overlap region in sensor coordinates
      double xoff_m = cameras.alt2xoff(alt) * cameras.pixel_sep;
      // compute distance map for each camera frame centered on the center of the overlap region
      double cx_L = 0; // FIXME do something here
      double cx_R = 0; // FIXME do something here
      interp::distance_map(left, alt, pitch, roll, width_m/2, height_m, cameras.focal_length, cx_L);
      interp::distance_map(right, alt, pitch, roll, width_m/2, height_m, cameras.focal_length, cx_R);
    } else {
      // for now, compute a distance map that in the case of stereo pairs spans the whole image
      interp::distance_map(D, alt, pitch, roll, width_m, height_m, cameras.focal_length);
    }
    // now discretize into slices
    int i = 0;
    Mat W = Mat::zeros(D.size(), CV_32F);
    while(cv::countNonZero(W) == 0) {
      interp::dist_weight(D, W, alt_step, i++);
    }
    while(cv::countNonZero(W) > 0) {
      Slice<int>* slice = getSlice(i);
      boost::mutex* mutex = slice->get_mutex();
      { // protect slice with mutex to prevent concurrent writes
	boost::lock_guard<boost::mutex> lock(*mutex);
	slice->getLightfield()->addImage(image, W);
      }
      interp::dist_weight(D, W, alt_step, i++);
    }
    return 0;
  }
  /**
   * Get the average image at the given altitude.
   * If the altitude is not located exactly at one of the altitude bins,
   * the average image is interpolated between any overlapping bins.
   * @param _dst the output image (zeros at desired resolution)
   * @param alt the altitude the image was taken at
   * @param pitch the pitch of the vehicle
   * @param roll the roll of the vehicle
   */
  void getAverage(cv::OutputArray _dst, double alt, double pitch, double roll) {
    Mat dst = _dst.getMat();
    int width = dst.size().width;
    int height = dst.size().height;
    Mat D = Mat::zeros(height, width, CV_32F);
    double width_m = width * cameras.pixel_sep;
    double height_m = height * cameras.pixel_sep;
    interp::distance_map(D, alt, pitch, roll, width_m, height_m, cameras.focal_length);
    // now discretize into slices
    int i = 0;
    Mat W = Mat::zeros(D.size(), CV_32F);
    while(cv::countNonZero(W) == 0) {
      interp::dist_weight(D, W, alt_step, i++);
    }
    while(cv::countNonZero(W) > 0) {
      Slice<int>* slice = getSlice(i);
      Mat sAverage;
      // get the slice average
      boost::mutex* mutex = slice->get_mutex();
      { // protect slice with mutex to prevent interleaved read/write operations
	boost::lock_guard<boost::mutex> lock(*mutex);
	sAverage = slice->getLightfield()->getAverage();
      }
      dst += sAverage.mul(W); // multiply by slice weight
      interp::dist_weight(D, W, alt_step, i++);
    }
  }
};
