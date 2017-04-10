#include <iostream>
#include "mlp.h"


int main() {
    float learningRate = 0.2;
    float momentum = 0.1;
    int nLayers = 3;
    int epochs = 10;
    int modelStruct[3] = {2,3,1};
    mlp neuralNet;
    neuralNet.create(modelStruct, nLayers, learningRate, momentum, epochs);
    int const nInputs = 2;
    float exemple[nInputs] = {0.12, 0.57};
    float *res = neuralNet.feedForward(exemple);
    std::cout << res[0] << std::endl;
    return 0;
}