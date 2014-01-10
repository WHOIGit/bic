#include <iostream>
#include <string>
#include <fstream>
#include <opencv2/opencv.hpp>

#include "demosaic.hpp"
#include "Lightfield.hpp"
#include "ThreadsafeQueue.hpp"

using namespace std;
using namespace cv;

// this is a prototype application; code is not in reusable state yet

// hardcoded input and output parameters
#define PATH_FILE "paths.txt"
#define MODEL_FILE "model.tiff"
#define OUT_DIR "out"
#define N_THREADS 8

// a learn job just specifies an input pathname
// or its stop flag is true, telling the worker to stop
class LearnJob {
public:
  string inpath;
  bool stop;
  LearnJob(String inp) {
    inpath = inp;
    stop = false;
  }
  LearnJob() {
    stop = true;
  }
};

// the learn worker accepts jobs from a queue and adds them to a lightfield
void learn_worker(Lightfield *model, ThreadsafeQueue<LearnJob>* queue) {
  static boost::mutex mutex; // shared lock for lightfield
  while(true) { // indefinitely,
    // atomically receive a job (this will block if the queue is empty,
    // and not return until jobs are pushed on the queue)
    LearnJob job = queue->pop();
    if(job.stop) { // is its stop flag true?
      cout << "DONE with one thread" << endl;
      return; // exit the thread
    } else {
      // get the input pathname
      string inpath = job.inpath;
      cout << "POPPED " << inpath << endl;
      // read the image (this can be done in parallel)
      Mat cfa_LR = imread(inpath, CV_LOAD_IMAGE_ANYDEPTH);
      cout << "Read " << inpath << endl;
      { // now lock the lightfield just long enough to add the image
	boost::lock_guard<boost::mutex> lock(mutex);
	model->addImage(cfa_LR);
      }
      cout << "Added " << inpath << endl;
    }
  }
}

// a correct job just has an input path and output path
class CorrectJob {
public:
  string inpath;
  string outpath;
  bool stop;
  CorrectJob(string inp, string outp) {
    inpath = inp;
    outpath = outp;
    stop = false;
  }
  CorrectJob() {
    stop = true;
  }
};

// the learn worker accepts jobs from a queue and corrects images
void correct_worker(Lightfield *model, ThreadsafeQueue<CorrectJob>* queue) {
  while(true) {
    // pop a job atomically
    CorrectJob job = queue->pop();
    if(job.stop) { // and stop if necessary
      cout << "DONE with one thread" << endl;
      return;
    }
    // no synchronization required as we are reading from the shared lighfield
    string inpath = job.inpath;
    string outpath = job.outpath;
    cout << "POPPED " << inpath << endl;
    Mat cfa_LR = model->correct(imread(inpath, CV_LOAD_IMAGE_ANYDEPTH));
    cout << "Demosaicing " << inpath << endl;
    Mat rgb_LR = demosaic(cfa_LR,"rgGb");
    cout << "Saving RGB to " << outpath << endl;
    Mat rgb_LR_16u;
    rgb_LR.convertTo(rgb_LR_16u, CV_16U);
    imwrite(outpath, rgb_LR_16u);
  }
}

// learn phase
void learn_prototype() {
  ifstream inpaths(PATH_FILE);
  string inpath;
  Lightfield model; // lightfield
  ThreadsafeQueue<LearnJob> work; // job queue
  boost::thread_group workers; // workers
  // first, push all the work onto the queue
  while(getline(inpaths,inpath)) { // read pathames from a file
    work.push(LearnJob(inpath)); // push the job
    cout << "PUSHED " << inpath << endl;
  }
  // now push a stop job per thread
  for(int i = 0; i < N_THREADS; i++) {
    work.push(LearnJob()); // tell thread to stop
  }
  // start up the work threads
  for(int i = 0; i < N_THREADS; i++) {
    boost::thread* worker = new boost::thread(learn_worker, &model, &work);
    workers.add_thread(worker); // add them to the thread group
  }
  // now join all threads. this will block until all threads have completed
  // all work.
  workers.join_all();
  // now checkpoint the lighfield
  cout << "Saving lightmap" << endl;
  model.save(MODEL_FILE);
  cout << "SUCCESS" << endl;
}

// correct application
void correct_prototype() {
  ifstream inpaths(PATH_FILE);
  string inpath;
  Lightfield model;
  // load the lightfield
  cout << "Loading lightmap" << endl;
  model.load(MODEL_FILE);
  int count = 0;
  ifstream inpaths2(PATH_FILE);
  // add all the jobs (see learn_prototype for how this works)
  ThreadsafeQueue<CorrectJob> work;
  boost::thread_group workers;
  for(int i = 0; i < N_THREADS; i++) {
    boost::thread* worker = new boost::thread(correct_worker, &model, &work);
    workers.add_thread(worker);
  }
  while(getline(inpaths2,inpath)) {
    stringstream outpaths;
    string outpath;
    outpaths << "out/correct" << count << ".tiff";
    outpath = outpaths.str();
    work.push(CorrectJob(inpath,outpath));
    cout << "PUSHED " << inpath << endl;
    count++;
  }
  for(int i = 0; i < N_THREADS; i++) {
    work.push(CorrectJob()); // tell thread to stop
  }
  workers.join_all();
  cout << "SUCCESS" << endl;
}
