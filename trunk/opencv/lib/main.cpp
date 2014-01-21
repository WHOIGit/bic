#include <string>
#include "stereo.hpp"
#include "prototype.hpp"

using namespace std;

int main(int argc, char **argv) {
  //xoff_test(argc,argv);
  if(string(argv[1])=="learn") {
    learn_prototype();
  } else {
    correct_prototype();
  }
}
