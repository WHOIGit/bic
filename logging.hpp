#pragma once
#include <sstream>
#include <boost/thread.hpp>
#include <boost/format.hpp>
#include <iostream>

namespace jlog {
  // adapted from http://stackoverflow.com/questions/1008019/c-singleton-design-pattern
  class log_mutex_ref {
  public:
    boost::mutex mutex;
    static log_mutex_ref& getInstance() {
      static log_mutex_ref instance;
      return instance;
    }
    boost::mutex* get_mutex() {
      return &mutex;
    }
  private:
    log_mutex_ref() {};
    log_mutex_ref(log_mutex_ref const&); // uncopyable
    void operator=(log_mutex_ref const&); // uncopyable
  };
  // adapted from http://stackoverflow.com/questions/1056411/how-to-pass-variable-number-of-arguments-to-printf-sprintf
  class formatted_log_t {
  public:
    formatted_log_t(const char* msg) : fmt(msg) {}
    ~formatted_log_t() {
      boost::mutex* mutex = log_mutex_ref::getInstance().get_mutex();
      { // protect stream from simultaneous output
	boost::lock_guard<boost::mutex> lock(*mutex);
	std::cerr << fmt << std::endl; // FIXME hardcoded destination stream
      }
    }
    
    template <typename T>
    formatted_log_t& operator %(T value) {
      fmt % value;
      return *this;
    }
  protected:
    boost::format fmt;
  };
  formatted_log_t log(const char *msg);
};
