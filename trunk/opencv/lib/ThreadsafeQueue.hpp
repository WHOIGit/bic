#include <queue>
#include <boost/thread.hpp>

// cribbed from
// http://thisthread.blogspot.com/2011/09/threadsafe-stdqueue.html
template <typename T>
class ThreadsafeQueue
{
private:
  std::queue<T> q_;
  boost::mutex m_; // 1
  boost::condition_variable c_;
 
public:
  void push(const T& data)
  {
    boost::lock_guard<boost::mutex> l(m_); // 1
    q_.push(data);
    c_.notify_one(); // 2
  }
 
  T pop()
  {
    boost::mutex::scoped_lock l(m_); // 3
    while(q_.empty()) {
	c_.wait(l); // 4
    }
 
    T res = q_.front();
    q_.pop();
    return res; // 5
  }
};

/* from http://thisthread.blogspot.com/2011/09/threadsafe-stdqueue.html:

1. We need just the bare RAII functionality from this lock, so
lock_guard is enough. Once the mutex is acquired, we can change the
object status, and that here means adding the passed data to the
queue.

2. We use the condition member variable to notify the other thread
that a new item has been inserted in the queue.

3. This lock should be passed to the member condition, in case we need
to wait for coming data, so a lock_guard is not enough.

4. If the queue is currently empty, we put the current thread in a
waiting status using the condition member variable. When the other
thread notify that something has changed, this thread would resume its
running status.  5. The value at the beginning of the queue is popped
and returned to the caller.
*/

