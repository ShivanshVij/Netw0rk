#include <iostream>
#include <cmath>
#include <cstdlib>

using namespace std;

int main(){
  //Random Training Sets for EXOR program (Two inputs, one output)

  cout << "topology: 2 4 1" << endl;
  for(int i = 2000; i >= 0; --i){
    int n1 = (int)(2.0 * rand() / double(RAND_MAX));
    int n2 = (int)(2.0 * rand() / double(RAND_MAX));
    int t = n1 ^ n2; // should be 0 or 1
    cout << "in: " << n1 << ".0 " << n2 << ".0 " << endl;
    if(t == 1){
      cout << "out: 0.0" << endl;
    }
    else{
      cout << "out: 1.0" << endl;
    }
  }
  return 0;
}
