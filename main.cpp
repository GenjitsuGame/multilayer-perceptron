#include <iostream>
#include "mlp.h"


int main() {
    float learningRate = 0.2;
    float momentum = 0.1;
    int nLayers = 3;
    int const epochs = 20000;
    int modelStruct[3] = {2, 4, 1};
    mlp neuralNet;
    neuralNet.create(modelStruct, nLayers, learningRate, momentum, epochs);
    int const nInputs = 4;
    int const nExemples = 4;
    float xorExemple[nExemples][nInputs] = {{0, 0},
                                            {0, 1},
                                            {1, 0},
                                            {1, 1}};
    float targetOutputs[nExemples][1] = {{0},
                                         {1},
                                         {1},
                                         {0}};
    for (int j = 0; j < epochs; ++j) {
        for (int i = 0; i < nExemples; ++i) {
            neuralNet.train(xorExemple[i], targetOutputs[i]);
        }
    }
    for (int k = 0; k < nExemples; ++k) {
        float *outputs = neuralNet.feedForward(xorExemple[k]);
        std::cout << outputs[0] << std::endl;
    }

    return 0;
}