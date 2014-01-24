#pragma once
#include <queue>
#include <boost/thread.hpp>

// cribbed from
// http://thisthread.blogspot.com/2011/09/threadsafe-stdqueue.html
template <typename T> class ThreadsafeQueue {
private:
  std::queue<T> q_;
  boost::mutex m_; // 1
  boost::condition_variable c_;
public:
  /**
   * Push an item on the queue, atomically
   * @param data the item to push on the queue
   */
  void push(const T& data) {
    boost::lock_guard<boost::mutex> l(m_); // protect write access to queue
    q_.push(data);
    c_.notify_one(); // notify waiting thread to wake up
  }
  /**
   * Remove an item from the queue and return it.
   * If the queue is empty, this method will block until an item
   * is added.
   */
  T pop() {
    // scoped lock so we can wait on the condition variable
    boost::mutex::scoped_lock l(m_);
    while(q_.empty()) {
	c_.wait(l); // release lock and wait for notification
    }
    // at this point the caller owns the lock again so we can safely
    // modify queue
    T res = q_.front();
    q_.pop();
    // after return the lock will fall out of scope and be released
    return res;
  }
};

