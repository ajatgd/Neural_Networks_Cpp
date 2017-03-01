#include <cstdlib>
#include <cassert>
#include <vector>
#include <cmath>
#include <iostream>
using namespace std;
struct Connection
{
  double weight;
  double deltaWeight;
}
class Neuron {};
typedef vector<Neuron> Layer;



class Neuron
{
public:
  Neuron(unsigned numOutputs, unsigned myIndex);
  void setOutputVal(double val) { m_outputVal = val;}
  double getOutputVal(void) const {return m_outputVal;}
  void feedForward(const Layer &prevLayer);
  void calcOutputGradients(double targetVal);
  void calcHiddenGradients(const Layer &nextLayer);
  void updateInputWeights(Layer &prevLayer);
private:
  static double eta;
  static double alpha;
  static double activationFunction(double x);
  static double activationFunctionDerivative(double x);
  static double randomWeight(void) {return rand() / double(RAND_MAX); }
  double m_outputVal;
  vector<Connection> m_outputWeights;
  unsigned m_myIndex;
  double m_gradient;
  double sumDOW(const Layer &nextLayer) const;
};
double Neuron::eta=0.15;//learning rate
double Neuron::alpha=0.5;

void Neuron::updateInputWeights(Layer &prevLayer)
{
  for(unsigned n=0;n<prevLayer.size();++n)
  {
    Neuron &neuron = prevLayer[n];
    double oldDeltaWeight=neuron.m_outputWeights[m_myIndex].deltaWeight;
    double newDeltaWeight=eta*neuron.getOutputVal()*m_gradient+alpha*oldDeltaWeight;//alpha is momentum
    neuron.m_outputWeights[m_myIndex].deltaWeight=newDeltaWeight;
    neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
  }
}
double Neuron::sumDOW(const Layer &nextLayer) const
{
  double sum=0.0;
  for(unsigned n=0;n<nextLayer.size()-1;++n)
  {
    sum+=m_outputWeights[n].weight*nextLayer[n].m_gradient;
  }
  return sum;
}
void Neuron::calcHiddenGradients(const Layer &nextLayer)
{
  double dow= sumDOW(nextLayer);
  m_gradient=dow*Neuron::activationFunctionDerivative(m_outputVal);
}
void Neuron::calcOutputGradients(double targetVal)
{
  double delta = targetVal-m_outputVal;
  m_gradient=delta*Neuron::activationFunctionDerivative(m_outputVal);
}
double Neuron::activationFunction(double x)
{
  return tanh(x);
}
double Neuron::activationFunctionDerivative(double x)
{
  return 1.0-x*x;
}
void Neuron::feedForward(const Layer &prevLayer)
{
  double sum=0.0;
  //add outputs of prevLayer including bias as they are input to current layer
  for(unsigned n=0; n< prevLayer.size(); ++n){
    sum+=prevLayer[n].getOutputVal() * prevLayer[n].m_outputWeights[m_myIndex].weight;
  }
  m_outputVal= Neuron::activationFunction(sum);
}
Neuron::Neuron(unsigned numOutputs, unsigned myIndex)
{
  for (unsigned c = 0; c< numOutputs; ++c)
  {
    m_outputWeights.push_back(Connection());
    m_outputWeights.back().weight = randomWeight();
  }
  m_myIndex=myIndex;

}



class Net
{
public:
  Net(const vector<unsigned> &topology);
  void feedForward(const vector<double> &inputVals) {};
  void backProp(const vector<double> &targetVals) {};
  void getResults(vector<double> &resultVals) const {};



private:
  vector<Layer> m_layers;
  double m_error;

};
void Net::getResults(vector<double> &resultVals) const
{
  resultVals.clear();
  for(unsigned n=0;n<m_layers.back().size()-1;++n)
  {
    resultVals.push_back(m_layers.back()[n].getOutputVal());
  }
}
void Net::backProp(const vector<double> &targetVals)
{
  //calculate errors
  Layer &outputLayer = m_layers.back();
  m_error = 0.0;

  for (unsigned n=0; n<outputLayer.size() -1; ++n)
  {
    double delta= targetVals[n]-outputLayer[n].getOutputVal();
    m_error /= outputLayer.size()-1;//avg errors
    m_error = sqrt(m_error);//RMS
    //calculating outputLayer gradient
    for (unsigned n = 0; n<outputLayer.size()-1; ++n)
    {
      outputLayer[n].calcOutputGradients(targetVals[n]);
    }
    //calculating gradients of hidden layer
    for(unsigned layerNum=m_layers.size()-2;layerNum>0; --layerNum){
      Layer &hiddenLayer = m_layers[layerNum];
      Layer &nextLayer=m_layers[layerNum+1];
      for(unsigned n=0;n<hiddenLayer.size();++n)
      {
        hiddenLayer[n].calcHiddenGradients(nextLayer);

      }
      //update weights
      for (unsigned layerNum = m_layers.size()-1; layerNum>0;--layerNum)
      {
        Layer &layer = m_layers[layerNum];
        Layer &prevLayer = m_layers[layerNum-1];
        for(unsigned n=0;n<layer.size()-1;++n)
        {
          layer[n].updateInputWeights(prevLayer);
        }
      }
    }

  }
}
void Net::feedForward(const vector<double> &inputVals)
{
  assert(inputVals.size() == n_layers[0].size()-1)//removing bias in count
  for (unsigned i=0; i< inputVals.size(); ++i)
  {
    m_layers[0][i].setOutputVal(inputVals[i]);

  }
  //forward propagate
  for(unsigned layerNum = 1; layerNum < n_layers.size(); ++layerNum)
  {
    Layer &prevLayer = m_layers[layerNum - 1];
    for(unsigned n = 0; n < m_layers[layerNum].size() - 1; ++n )
    {
      m_layers[layerNum][n].feedForward(prevLayer);
    }
  }
}

Net::Net(const vector<unsigned> &topology)
{
  unsigned numLayers = topology.size();
  for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum){
    m_layers.push_back(Layer());
    unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];
    //adding neurons to new layer with an extra bias term
    for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum){
      m_layers.back().push_back(Neuron(numOutputs,neuronNum));
      cout<< "Made a neuron!"<<endl;
      m_layers.back().setOutputVal(1.0);
    }
  }
}

int main()
{
  vector<unsigned> topology;
  topology.push_back(3);
  topology.push_back(2);
  topology.push_back(1);
  Net myNet(topology);
  vector<double> inputVals;
  myNet.feedForward(inputVals);
  vector<double> targetVals;
  myNet.backProp(targetVals);
  vector<double> resultVals;
  myNet.getResults(resultVals);
}
