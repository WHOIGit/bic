#pragma once
#include <sstream>
#include <boost/thread.hpp>
#include <boost/format.hpp>
#include <iostream>
#include <boost/date_time/posix_time/posix_time.hpp>

namespace jlog {
  // adapted from http://stackoverflow.com/questions/1008019/c-singleton-design-pattern
  class log_mutex_ref { // singleton mutex for logging
  public:
    boost::mutex mutex; // the actual mutex
    static log_mutex_ref& getInstance() { // factory method
      static log_mutex_ref instance;
      return instance;
    }
    boost::mutex* get_mutex() { // mutex accessor
      return &mutex;
    }
  private:
    log_mutex_ref() {}; // unconstructable
    log_mutex_ref(log_mutex_ref const&); // uncopyable
    void operator=(log_mutex_ref const&); // uncopyable
  };
  // adapted from http://stackoverflow.com/questions/1056411/how-to-pass-variable-number-of-arguments-to-printf-sprintf
  class formatted_log_t {
  private:
    std::ostream* os; // stream to write to
  public:
    formatted_log_t(const char* msg, std::ostream* _os=&std::cout) : fmt(msg) {
      os=_os;
    }
    ~formatted_log_t() {
      using namespace boost::posix_time;
      ptime tm = microsec_clock::universal_time();
      boost::mutex* mutex = log_mutex_ref::getInstance().get_mutex();
      { // protect stream from simultaneous output
	boost::lock_guard<boost::mutex> lock(*mutex);
	*os << to_iso_extended_string(tm) << "Z," << fmt << std::endl;
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
  /** Log a message to cout. Supports the '%' operator for variable interpolation */
  formatted_log_t log(const char *msg);
  /** Log a message to cerr. Supports the '%' operator for variable interpolation */
  formatted_log_t log_error(const char *msg);
};
