#pragma once
#include <cmath>

namespace interp {
  template <typename T> class LinearBinning;
}

template <typename T> class interp::LinearBinning {
private:
  std::vector<T> bins;
public:
  LinearBinning() { }
  LinearBinning(T low, T high, T width) {
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
