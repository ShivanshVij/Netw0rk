#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>
#include <iterator>
#include <string>

using namespace std;

/*          TrainingData Class START         */
class TrainingData{
public:
    TrainingData(const string filename);
    bool isEof(void) { return neurontrainingDataFile.eof(); }
    void getTopology(vector<unsigned> &topology);

    // Returns the number of input values read from the file:
    unsigned getNextInputs(vector<double> &inputValues);
    unsigned getTargetOutputs(vector<double> &targetOuputValues);
private:
    ifstream neurontrainingDataFile;
};
void TrainingData::getTopology(vector<unsigned> &topology){
    string line;
    string label;

    getline(neurontrainingDataFile, line);
    stringstream ss(line);
    ss >> label;
    if (this->isEof() || label.compare("topology:") != 0){
        abort();
    }

    while(!ss.eof()){
        unsigned n;
        ss >> n;
        topology.push_back(n);
    }

    return;
}
TrainingData::TrainingData(const string filename){
    neurontrainingDataFile.open(filename.c_str());
}
unsigned TrainingData::getNextInputs(vector<double> &inputValues){
    inputValues.clear();

    string line;
    getline(neurontrainingDataFile, line);
    stringstream ss(line);

    string label;
    ss>> label;
    if (label.compare("in:") == 0){
        double oneValue;
        while(ss >> oneValue){
            inputValues.push_back(oneValue);
        }
    }

    return inputValues.size();
}
unsigned TrainingData::getTargetOutputs(vector<double> &targetOutputValues){
    targetOutputValues.clear();

    string line;
    getline(neurontrainingDataFile, line);
    stringstream ss(line);

    string label;
    ss>> label;
    if (label.compare("out:") == 0){
        double oneValue;
        while(ss >> oneValue){
            targetOutputValues.push_back(oneValue);
        }
    }

    return targetOutputValues.size();
}
/*          TrainingData Class END         */

/*          Connection Stucture START         */
struct Connection{
  double weight;
  double deltaWeight;
};
/*          Connection Stucture END         */

/*          Neuron Class START         */
class Neuron; //Define Neuron here first so that the typedef for Layer exists
typedef vector<Neuron> Layer;
class Neuron{
public:
  //Public functions
  Neuron(unsigned numberofOutputs, unsigned Index);
  void setOutputValue(double value) { neuronOutputValue = value; }
  double getOutputValue(void) const { return neuronOutputValue; }
  void feedForward(const Layer &prevLayer);
  void calculateOutputGradients(double targetValues);
  void calculateHiddenGradients(const Layer &nextLayer);
  void updateInputWeights(Layer &prevLayer);
private:
  //Private functions
  static double eta;
  static double alpha;
  static double transferFunction(double x);
  static double transferFunctionDerivative(double x);
  static double randomWeight(void) { return rand() / double(RAND_MAX); }
  double sumDOW(const Layer &nextLayer) const;
  double neuronOutputValue;
  vector<Connection> neuronOutputWeights;
  unsigned neuronIndex;
  double neuronGradient;
};
double Neuron::eta = 0.15; //Overall net learning rate
double Neuron::alpha = 0.5; //multiplier of last deltaWeight
void Neuron::updateInputWeights(Layer &prevLayer){
  for(unsigned n = 0; n < prevLayer.size(); ++n){
    Neuron &neuron = prevLayer[n];
    double oldDeltaWeight = neuron.neuronOutputWeights[neuronIndex].deltaWeight;

    double newDeltaWeight =
              eta
              * neuron.getOutputValue()
              * neuronGradient
              + alpha
              *oldDeltaWeight;

    neuron.neuronOutputWeights[neuronIndex].deltaWeight = newDeltaWeight;
    neuron.neuronOutputWeights[neuronIndex].weight += newDeltaWeight;
  }
}
double Neuron::sumDOW(const Layer &nextLayer) const{
  double sum = 0.0;
  for(unsigned n = 0; n < nextLayer.size(); ++n){
    sum += neuronOutputWeights[n].weight * nextLayer[n].neuronGradient;
  }
  return sum;
}
void Neuron::calculateHiddenGradients(const Layer &nextLayer){
  double dow = sumDOW(nextLayer);
  neuronGradient = dow*Neuron::transferFunctionDerivative(neuronOutputValue);
}
void Neuron::calculateOutputGradients(double targetValues){
  double delta = targetValues - neuronOutputValue;
  neuronGradient = delta*Neuron::transferFunctionDerivative(neuronOutputValue);
}
double Neuron::transferFunction(double x){
  //tanh [0 to 1] {FOR XOR or EXOR Training Data}
  //return tanh(x);

  //linear [-infinity to infinity] {For Addition or Subtraction Training Data}
  return 0.01*x;

}
double Neuron::transferFunctionDerivative(double x){
  //tanh derivative [0 to 1] {FOR XOR or EXOR Training Data}
  //return 1.0-x*x;

  //linear [-infinity to infinity] {For Addition or Subtraction Training Data}
  return 0.01;
}
void Neuron::feedForward(const Layer &prevLayer){
  double sum = 0.0;

  //Sum the previous layer's outpouts (which will be our inputs)
  //And also include the bias node from the previous Layer
  for(unsigned n = 0; n < prevLayer.size(); ++n){
    sum += prevLayer[n].getOutputValue() *
            prevLayer[n].neuronOutputWeights[neuronIndex].weight;
  }

  neuronOutputValue = Neuron::transferFunction(sum);
}
Neuron::Neuron(unsigned numberofOutputs, unsigned Index){
  for(unsigned c = 0; c < numberofOutputs; ++c){
    neuronOutputWeights.push_back(Connection());
    neuronOutputWeights.back().weight = randomWeight(); //Assign a random weight to each Connection structure in each 'Neuron'
  }

  neuronIndex = Index;
}
/*          Neuron Class END         */

/*          Network Class START        */
class Network{
public:
  //Public functions
  Network(const vector<unsigned> &topology);
  void feedForward(const vector<double> &inputValues); //feedForward simply reads the inputValues and does not change them
  void backPropogate(const vector<double> &targetValues); //backPropogate simple reads the targetValues for training and does not change them
  void getResults(vector<double> &resultValues) const; //getResults simply reads the resultValues and never modifies them, thus it is a const however the vector resultValues does get changed thus it is not a const
  double getRecentAverageError(void) const { return neuronRecentAverageError; }
private:
  //Private functions
  vector<Layer> neuronLayer; //neuronLayer[layerNumber][neuronNumber]
  double neuronError;
  double neuronRecentAverageError;
  double neuronRecentAverageSmoothingFactor;
};
void Network::getResults(vector<double> &resultValues) const{
  resultValues.clear();
  for(unsigned n = 0; n < neuronLayer.back().size() - 1; ++n){
    resultValues.push_back(neuronLayer.back()[n].getOutputValue());
  }
}
void Network::backPropogate(const vector<double> &targetValues){
  //Calculate overall net error (Root Mean Square Error of Output Neuron Errors)
  Layer &outputLayer = neuronLayer.back();
  neuronError = 0.0;

  for(unsigned n = 0; n < outputLayer.size() - 1; ++n){
    double delta = targetValues[n] - outputLayer[n].getOutputValue();
    neuronError += delta*delta;
  }
  neuronError = sqrt(neuronError);

  //Implement a recent average measurement:
  neuronRecentAverageError =
              (neuronRecentAverageError * neuronRecentAverageSmoothingFactor + neuronError)
              / (neuronRecentAverageSmoothingFactor + 1.0);

  //Calculate ouput layer gradients
  for(unsigned n = 0; n < outputLayer.size() - 1; ++n){
    outputLayer[n].calculateOutputGradients(targetValues[n]);
  }

  //Calculate gradients on hidden layers
  for(unsigned layerNumber = neuronLayer.size() - 2; layerNumber > 0; --layerNumber){
    Layer &hiddenLayer = neuronLayer[layerNumber];
    Layer &nextLayer = neuronLayer[layerNumber + 1];

    for(unsigned n = 0; n < hiddenLayer.size(); ++n){
      hiddenLayer[n].calculateHiddenGradients(nextLayer);
    }
  }

  //For all layers from outputs to first hiden layer, update connection weights
  for(unsigned layerNumber = neuronLayer.size() - 1; layerNumber > 0; --layerNumber){
    Layer &layer = neuronLayer[layerNumber];
    Layer &prevLayer = neuronLayer[layerNumber - 1];

    for(unsigned n = 0; n < layer.size() - 1; ++n){
      layer[n].updateInputWeights(prevLayer);
    }
  }

}
void Network::feedForward(const vector<double> &inputValues){
  //assert(inputValues.size() == neuronLayer[0].size() - 1); //Tell the program what we think should happen, and if an error occurs the user will be notified

  for(unsigned i = 0; i < inputValues.size(); ++i){ //Attach the input values to the input neurons
    neuronLayer[0][i].setOutputValue(inputValues[i]);
  }

  for(unsigned layerNumber = 1; layerNumber < neuronLayer.size(); ++layerNumber){
    Layer &prevLayer = neuronLayer[layerNumber - 1];
    for(unsigned n = 0; n < neuronLayer[layerNumber].size() - 1; ++n){
      neuronLayer[layerNumber][n].feedForward(prevLayer);
    }
  }
}
Network::Network(const vector<unsigned> &topology){
  unsigned numberofLayers = topology.size();

  for(unsigned layerNumber = 0; layerNumber < numberofLayers; ++layerNumber){
    neuronLayer.push_back(Layer()); //Make a new layer, put it onto the end of neuronLayer
    unsigned numberofOutputs = layerNumber == topology.size() - 1 ? 0 : topology[layerNumber + 1];

    for(unsigned neuronNumber = 0; neuronNumber <= topology[layerNumber]; ++neuronNumber){//We want to add an extra biased neuron so it's "<="
      neuronLayer.back().push_back(Neuron(numberofOutputs, neuronNumber));
      cout << "Neuron Created" << endl;
    }
    neuronLayer.back().back().setOutputValue(1.0);
  }
}
void showVectorValues(string label, vector<double> &v){
    cout << label << " ";
    for(unsigned i = 0; i < v.size(); ++i){
        cout << v[i] << " ";
    }

    cout << endl;
}
/*          Network Class END        */

/*          main function START        */
int main(){

  printf("%s", "Enter Path to Training File: ");

  string TrainingDataFileName;
  cin >> TrainingDataFileName;

  TrainingData trainData(TrainingDataFileName);

  vector<unsigned> topology;
  trainData.getTopology(topology);

  Network theNetwork(topology); //construct the Neural Network object from the "Network" class

  vector<double> inputValues;
  vector<double> targetValues;
  vector<double> resultValues;

  int trainingPass = 0;

  while(!trainData.isEof()){
      ++trainingPass;
      cout << endl << "Pass " << trainingPass;

      // Get new input data and feed it forward:
      if(trainData.getNextInputs(inputValues) != topology[0]){
          break;
      }
      showVectorValues(": Inputs:", inputValues);
      theNetwork.feedForward(inputValues);

      // Collect the net's actual output results:
      theNetwork.getResults(resultValues);
      showVectorValues("Outputs:", resultValues);

      // Train the net what the outputs should have been:
      trainData.getTargetOutputs(targetValues);
      showVectorValues("Targets:", targetValues);
      //assert(targetValues.size() == topology.back());

      theNetwork.backPropogate(targetValues);

      // Report how well the training is working, average over recent samples:
      cout << "Net recent average error: "
              << theNetwork.getRecentAverageError() << endl;
  }

  cout << endl << "Training is Complete. Please enter your own data now." << endl << endl;

  cout << "Number of Trials you would like to perform: ";

  int numberofTrials = 0;

  cin >> numberofTrials;

  for(int TRIALS = 0; TRIALS < numberofTrials; ++TRIALS){
    cout << "Enter Data: ";
    int input1, input2;
    cin >> input1 >> input2;
    inputValues.clear();
    inputValues.push_back(input1);
    inputValues.push_back(input2);

    theNetwork.feedForward(inputValues);

    theNetwork.getResults(resultValues);

    string answerString;
    ostringstream oss;

    if (!resultValues.empty())
    {
      // Convert all but the last element to avoid a trailing ","
      copy(resultValues.begin(), resultValues.end()-1,
          ostream_iterator<int>(oss, ","));

      // Now add the last element with no delimiter
      oss << resultValues.back();
    }

    answerString = oss.str();

    double answerDouble = atof(answerString.c_str());

    //cout << "String is: " << answerString << endl;

    int answerInt = (int)round(answerDouble);

    cout << "Output: " << answerInt << endl;

    /*if(answerDouble < 0.5){
      cout << "Output: 0" << endl;
    }
    else{
      cout << "Output: 1" << endl;
    }*/

    //showVectorValues("Outputs:", resultValues);

  }

  cout << endl << "Done" << endl;

  return 0;
}
/*          main function END        */
