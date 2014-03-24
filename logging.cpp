#include "logging.hpp"

jlog::formatted_log_t jlog::log(const char *msg) {
  return jlog::formatted_log_t(msg);
}
