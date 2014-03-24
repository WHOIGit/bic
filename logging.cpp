#include "logging.hpp"

jlog::formatted_log_t jlog::log(const char *msg) {
  return jlog::formatted_log_t(msg);
}

jlog::formatted_log_t jlog::log_error(const char *msg) {
  return jlog::formatted_log_t(msg, &std::cerr);
}
