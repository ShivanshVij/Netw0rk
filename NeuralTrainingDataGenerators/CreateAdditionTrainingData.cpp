#include <iostream>
#include <cmath>
#include <cstdlib>

using namespace std;

int main(){
  //Random Training Sets for EXOR program (Two inputs, one output)

  cout << "topology: 2 10 1" << endl;
  for(int i = 2000; i >= 0; --i){
    int n1 = rand() % 100;
    int n2 = rand() % 100;
    int t = n1 + n2; // should be 0 or 1
    cout << "in: " << n1 << ".0 " << n2 << ".0 " << endl;
    cout << "out: " << t << ".0" << endl;
  }
  return 0;
}
