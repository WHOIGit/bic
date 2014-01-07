#include <iostream>
#include <cstdlib>
#include <pthread.h>
#include <time.h>
#include <unistd.h>
#include <vector>
#include "Pthreads.hpp"
#include "pthreads_example.hpp"

using namespace std;

void *announce_records(void *l) {
	Lockable<int> *record = static_cast<Lockable<int> *>(l);
    cout << "Announcer acquiring monitor..." << endl;
	record->lock();
    cout << "Announcer got monitor" << endl;
	int current = 0;
	while(current < 999) {
		record->wait();
        int newCurrent = record->get();
        if(newCurrent > current) {
            cout << "Announcer woke up" << endl;
            current = newCurrent;
            cout << "New record: " << current << endl;
        }
	}
	cout << "Winning number: " << current << endl;
	record->unlock();
	pthread_exit(NULL);
}

void *generate_candidates(void *l) {
	Lockable<int> *record = static_cast<Lockable<int> *>(l);
    cout << "In a generation thread" << endl;
	int current = 0;
	do {
		sleep((rand() % 5) + 1);
        cout << "Generator unsleeping, acquiring monitor" << endl;
		record->lock();
		int candidate = (rand() % 600) + (rand() % 600);
		cout << "Generator got lock and generated " << candidate << endl;
		current = record->get();
		if(candidate > current) {
			record->set(candidate);
            cout << "Generator waking up announcer..." << endl;
			record->broadcast();
		}
        cout << "Generator releasing monitor..." << endl;
		record->unlock();
	} while(current < 999);
	pthread_exit(NULL);
}

void run_example() {
	// in this example application, several threads simultaneously
	// compete to produce the highest random number, and there is
	// a console notification when each new record is achieved.

	// initialize stuff
	srand(time(NULL)); // random number generator
	
	// create the lockable record number
	Lockable<int> recordLockable = Lockable<int>(0);
    void *record = static_cast<void *>(&record);

    // for some unexplained reason, this deadlocks sometimes.
	Runnable announcer(announce_records);
	announcer.start(record);
    
	// start other threads
    int N = 4;
	Runnable generators[4];
    for(int i = 0; i < N; i++) {
        generators[i].setRunner(generate_candidates);
        generators[i].start(record);
    }
    
    for(int i = 0; i < N; i++) {
        cout << "Joining generator thread " << i << endl;
        generators[i].join();
        cout << "Joined generator thread " << i << endl;
    }
    
    cout << "Joining announcer thread" << endl;
    announcer.join();
    cout << "Joined announcer thread" << endl;
}
