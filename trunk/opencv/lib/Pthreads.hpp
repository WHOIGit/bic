#pragma once
#include <pthread.h>

using namespace std;

// provides a mutex and single monitor (condition) associated
// with an aribtrary data object, as well as setters and getters.
// callers are responsible for using locking and signalling
// correctly.
template <typename T> class Lockable {
    pthread_mutex_t mutex;
    pthread_cond_t monitor;
    T lockable;
public:
    Lockable(T object) {
        pthread_mutex_init(&mutex, NULL);
        pthread_cond_init(&monitor, NULL);
        lockable = object;
    };
    T get() {
        return lockable;
    };
    void set(T object) {
        lockable = object;
    };
    void lock() {
        pthread_mutex_lock(&mutex);
    };
    void unlock() {
        pthread_mutex_unlock(&mutex);
    };
    void signal() {
        pthread_cond_signal(&monitor);
    };
    void broadcast() {
        pthread_cond_broadcast(&monitor);
    };
    void wait() {
        pthread_cond_wait(&monitor, &mutex);
    };
};

// wraps pthread create and join for runnability
class Runnable {
    pthread_t thread;
    pthread_attr_t pattr;
    void *(*runner)(void *);
    void init() {
        pthread_attr_init(&pattr);
        pthread_attr_setdetachstate(&pattr, PTHREAD_CREATE_JOINABLE);
    }
public:
    Runnable() {
        init();
    }
    Runnable(void *(*run)(void *)) {
        init();
        runner = run;
    }
    void setRunner(void *(*run)(void *)) {
        runner = run;
    }
    void start() {
        pthread_create(&thread, &pattr, runner, NULL);
    }
    void start(void *in_data) {
        pthread_create(&thread, &pattr, runner, in_data);
    }
    void join() {
        pthread_join(thread, NULL);
    }
};
