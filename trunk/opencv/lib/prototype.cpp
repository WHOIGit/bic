#include <iostream>
#include <string>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <boost/tokenizer.hpp>

#include "demosaic.hpp"
#include "illumination.hpp"
#include "threadutils.hpp"

using namespace std;
using namespace cv;

using illum::MultiLightfield;

// this is a prototype application; code is not in reusable state yet

// hardcoded input and output parameters
#define PATH_FILE "alts.csv"
#define MODEL_FILE "model.tiff"
#define OUT_DIR "out"
#define N_THREADS 12

// a learn job just specifies an input pathname and an altitude
// or its stop flag is true, telling the worker to stop
class LearnJob {
public:
  string inpath;
  int altitude;
  bool stop;
  LearnJob(String inp, float alt=150) {
    inpath = inp;
    altitude = alt;
    stop = false;
  }
  LearnJob() {
    stop = true;
  }
};

// the learn worker accepts jobs from a queue and adds them to a lightfield
void learn_worker(MultiLightfield<int> *model, AsyncQueue<LearnJob>* queue) {
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
      cout << "POPPED " << inpath << " " << job.altitude << endl;
      // read the image (this can be done in parallel)
      Mat cfa_LR = imread(inpath, CV_LOAD_IMAGE_ANYDEPTH);
      cout << "Read " << inpath << endl;
      { // now lock the lightfield just long enough to add the image
	boost::lock_guard<boost::mutex> lock(mutex);
	model->addImage(cfa_LR, job.altitude);
      }
      cout << "Added " << inpath << endl;
    }
  }
}

// a correct job just has an input path and output path
// and an altitude
class CorrectJob {
public:
  string inpath;
  string outpath;
  int altitude;
  bool stop;
  CorrectJob(string inp, string outp, int alt=150) {
    inpath = inp;
    outpath = outp;
    altitude = alt;
    stop = false;
  }
  CorrectJob() {
    stop = true;
  }
};

// the learn worker accepts jobs from a queue and corrects images
void correct_worker(MultiLightfield<int> *model, AsyncQueue<CorrectJob>* queue) {
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
    cout << "POPPED " << inpath << " " << job.altitude << endl;
    Mat average = model->getAverage(job.altitude);
    Mat cfa_LR = imread(inpath, CV_LOAD_IMAGE_ANYDEPTH);
    illum::correct(cfa_LR, cfa_LR, average);
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
  string line;
  MultiLightfield<int> model(100, 300, 10); // lightfield

  // learning
  AsyncQueue<LearnJob> lwork; // job queue
  boost::thread_group lworkers; // workers
  // first, push all the work onto the queue
  typedef boost::tokenizer< boost::escaped_list_separator<char> > Tokenizer;
  vector<string> fields;
  while(getline(inpaths,line)) { // read pathames from a file
    Tokenizer tok(line);
    fields.assign(tok.begin(),tok.end());
    string inpath = fields.front();
    int alt = (int)(atof(fields.back().c_str()) * 100);
    lwork.push(LearnJob(inpath,alt)); // push the job
    cout << "PUSHED " << inpath << endl;
  }
  // now push a stop job per thread
  for(int i = 0; i < N_THREADS; i++) {
    lwork.push(LearnJob()); // tell thread to stop
  }
  // start up the work threads
  for(int i = 0; i < N_THREADS; i++) {
    boost::thread* worker = new boost::thread(learn_worker, &model, &lwork);
    lworkers.add_thread(worker); // add them to the thread group
  }
  // now join all threads. this will block until all threads have completed
  // all work.
  lworkers.join_all();
  cout << "SUCCESS learn phase" << endl;

  model.save();
  cout << "SAVED model" << endl;

  // add all the jobs (see learn_prototype for how this works)
  AsyncQueue<CorrectJob> cwork;
  boost::thread_group cworkers;
  for(int i = 0; i < N_THREADS; i++) {
    boost::thread* worker = new boost::thread(correct_worker, &model, &cwork);
    cworkers.add_thread(worker);
  }
  ifstream inpaths2(PATH_FILE);
  int count = 0;
  while(getline(inpaths2,line)) {
    // FIXME only correcting every 5th image
    if(count % 5 == 0) {
      Tokenizer tok(line);
      fields.assign(tok.begin(),tok.end());
      string inpath = fields.front();
      int alt = (int)(atof(fields.back().c_str()) * 100);
      stringstream outpaths;
      string outpath;
      outpaths << OUT_DIR << "/correct" << count << ".tiff";
      outpath = outpaths.str();
      cwork.push(CorrectJob(inpath,outpath,alt));
      cout << "PUSHED " << inpath << endl;
    }
    count++;
  }
  for(int i = 0; i < N_THREADS; i++) {
    cwork.push(CorrectJob()); // tell thread to stop
  }
  cworkers.join_all();
  cout << "SUCCESS correct phase" << endl;
}
